import logging
import os, io

import torch
from PIL import Image
import torchvision.models as models
from torchvision import transforms
from torch.nn import (
    AdaptiveAvgPool2d,
    AdaptiveMaxPool2d,
    Flatten,
    BatchNorm1d,
    Dropout,
    Linear,
    ReLU,
    Module,
    Sequential,
)

logger = logging.getLogger(__name__)


class HookCAM:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_func)
        self.stored = []

    def hook_func(self, module, inputs, outputs):
        self.stored.append(outputs.detach().clone())

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.hook.remove()


class HookCAMBwd:
    def __init__(self, module):
        self.hook = module.register_full_backward_hook(self.hook_func)
        self.stored = []

    def hook_func(self, module, grad_inputs, grad_outputs):
        self.stored.append(grad_outputs[0].detach().clone())

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.hook.remove()


class AdaptiveConcatPool2d(Module):
    "FastAI: Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`"

    def __init__(self, size=None):
        super().__init__()
        self.size = size or 1
        self.ap = AdaptiveAvgPool2d(self.size)
        self.mp = AdaptiveMaxPool2d(self.size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class TwinHandler:
    """
    Handler Class.
    """

    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False

        self.encoder_reload, self.head_input_dim = self.get_encoder()
        self.head_reload = Sequential(
            AdaptiveConcatPool2d(1),
            Flatten(),
            BatchNorm1d(self.head_input_dim),
            Dropout(0.05),
            Linear(self.head_input_dim, 512, False),
            ReLU(True),
            BatchNorm1d(512),
            Dropout(0.1),
            Linear(512, 2, False),
        )
        self.image_tfm = transforms.Compose(
            [
                # must be consistent with model training
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # default statistics from imagenet
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.cam: bool = False

    @staticmethod
    def get_encoder(pre_train: bool = False):
        resnet_backbone = models.resnet50(pretrained=pre_train)
        return Sequential(*list(resnet_backbone.children())[:-2]), 8192

    def initialize(self, ctx):
        """
        load eager mode state_dict based model
        """
        properties = ctx.system_properties
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )

        logger.info(f"Device on initialization is: {self.device}")
        model_dir = properties.get("model_dir")

        manifest = ctx.manifest
        logger.error(manifest)
        serialized_file = manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model definition file")

        logger.info(model_pt_path)
        encoder_reload_weights = torch.load(
            os.path.join(model_dir, "encoder_weight.pth"), map_location=self.device
        )
        self.encoder_reload.load_state_dict(encoder_reload_weights)
        self.encoder_reload.to(self.device)
        self.encoder_reload.eval()

        head_reload_weights = torch.load(
            os.path.join(model_dir, "head_weight.pth"), map_location=self.device
        )
        self.head_reload.load_state_dict(head_reload_weights)
        self.head_reload.to(self.device)
        self.head_reload.eval()

        self.initialized = True

    def preprocess(self, data):
        """
        Scales and normalizes a PIL image for an U-net model
        """
        self.cam_map_left, self.cam_map_right = None, None
        left_image = data[0]["left"]
        right_image = data[0]["right"]
        self.cam = eval(data[0]["cam"])
        logger.info(f"input cam: {str(self.cam)}")

        left_image = Image.open(io.BytesIO(left_image)).convert("RGB")
        right_image = Image.open(io.BytesIO(right_image)).convert("RGB")

        left_image = self.image_tfm(left_image)[None, ...]  # batch size of 1
        right_image = self.image_tfm(right_image)[None, ...]  # batch size of 1

        return left_image, right_image

    def inference(self, left_image, right_image):
        """
        Predict the chip stack mask of an image using a trained deep learning model.
        """
        if not self.cam:
            logger.info("no cam")
            with torch.no_grad():
                left_image, right_image = left_image.to(self.device), right_image.to(
                    self.device
                )
                left_embedding = self.encoder_reload(left_image)
                right_embedding = self.encoder_reload(right_image)
                res = self.head_reload(
                    torch.cat([left_embedding, right_embedding], dim=1)
                )[0]

        else:
            logger.info("cam")
            with HookCAMBwd(self.encoder_reload) as hookg:
                with HookCAM(self.encoder_reload) as hook:
                    left_image, right_image = left_image.to(
                        self.device
                    ), right_image.to(self.device)
                    left_embedding = self.encoder_reload(left_image)
                    right_embedding = self.encoder_reload(right_image)

                    res = self.head_reload(
                        torch.cat([left_embedding, right_embedding], dim=1)
                    )[0]
                    act = hook.stored

                pred_cls = res.argmax().item()
                res[pred_cls].backward()
                grad = hookg.stored
            
            weight_left = grad[0][0].mean(dim=[1, 2], keepdim=True)
            self.cam_map_left = (weight_left * act[0][0]).sum(0)

            weight_right = grad[1][0].mean(dim=[1, 2], keepdim=True)
            self.cam_map_right = (weight_right * act[1][0]).sum(0)
            
            self.encoder_reload.zero_grad(), self.head_reload.zero_grad()

        return res

    def postprocess(self, inference_output):
        logger.info("start postprocessing")
        if torch.cuda.is_available():
            inference_output = inference_output.cpu()
            if self.cam:
                self.cam_map_left = self.cam_map_left.cpu()
                self.cam_map_right = self.cam_map_right.cpu()
        else:
            inference_output = inference_output

        if not self.cam:
            return [inference_output.numpy().tolist()]
        else:
            return [
                inference_output.detach().numpy().tolist()
                + self.cam_map_left.detach().numpy().tolist()
                + self.cam_map_right.detach().numpy().tolist()
            ]


_service = TwinHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    left_image, right_image = _service.preprocess(data)
    data = _service.inference(left_image, right_image)
    data = _service.postprocess(data)

    return data
