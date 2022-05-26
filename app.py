import streamlit as st
from io import BytesIO
from PIL import Image
import requests
import boto3
from math import exp
import matplotlib.pyplot as plt


SAGEMAKER_ENDPOINTS = ["XXXXXXX"]
REGIONS = ["XXXXXX"]


def img_upload_side_bar(col_display) -> tuple:
    def upload_vis_image(file_path, col_vis):
        if file_path is not None:
            image = (
                Image.open(BytesIO(file_path.read()), mode="r")
                .convert("RGB")
                .resize((600, 400))
            )
            with col_vis:
                image_loc = st.empty()
            image_loc.image(image, use_column_width=True)
            return image_loc  # col_vis.image(image, use_column_width=True)

    st.sidebar.title("Image Uploading")
    # Disabling warning
    st.set_option("deprecation.showfileUploaderEncoding", False)
    # Choose your own image
    left_file = st.sidebar.file_uploader(
        "Upload 1st Image", type=["png", "jpeg", "jpg"]
    )
    left_image_loc = upload_vis_image(left_file, col_display[0])

    right_file = st.sidebar.file_uploader(
        "Upload 2nd Image", type=["png", "jpeg", "jpg"]
    )
    right_image_loc = upload_vis_image(right_file, col_display[1])

    return left_file, right_file, left_image_loc, right_image_loc


def button(left_image_path, right_image_path):
    st.sidebar.write("\n")
    st.sidebar.title("Model Endpoint")

    region = st.sidebar.selectbox("AWS Region", REGIONS, help="AWS Region")
    endpoint_name = st.sidebar.selectbox(
        "Amazon SageMaker Endpoint",
        SAGEMAKER_ENDPOINTS,
        help="Amazon SageMaker Endpoint to invoke",
    )
    cam = st.sidebar.checkbox("Classification Activation Map (CAM)")

    if st.sidebar.button("Predict"):
        if left_image_path is None:
            st.sidebar.error("Please upload 1st image.")
            return None, None, None

        if right_image_path is None:
            st.sidebar.error("Please upload 2nd image.")
            return None, None, None

        elif left_image_path is not None and right_image_path is not None:
            with st.spinner("Analyzing..."):
                r = requests.Request(
                    "POST",
                    "http://localhost:8080/invocations",
                    data={"cam": str(cam)},
                    files={
                        "left": left_image_path.getvalue(),
                        "right": right_image_path.getvalue(),
                    },
                ).prepare()
                content_type, payload = r.headers["Content-Type"], r.body

                client = boto3.client("sagemaker-runtime", region_name=region)
                response = client.invoke_endpoint(
                    EndpointName=endpoint_name, ContentType=content_type, Body=payload
                )
                neg, pos, *maps = eval(response["Body"].read())
            return neg, pos, maps
    else:
        return None, None, None


def prediction(negative_logit, positive_logit):
    return exp(negative_logit) / (exp(negative_logit) + exp(positive_logit)), exp(
        positive_logit
    ) / (exp(negative_logit) + exp(positive_logit))


def cam_vis(cam_map, col_vis, file_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_visible(False)
    image = (
        Image.open(BytesIO(file_path.getvalue()), mode="r")
        .convert("RGB")
        .resize((600, 400))
    )
    ax.imshow(image)
    ax.imshow(
        cam_map,
        alpha=0.3,
        extent=(0, 600, 400, 0),
        interpolation="bilinear",
        cmap="jet",
    )
    ax.axis("off")
    ax.set(frame_on=False)
    fig.tight_layout()
    col_vis.pyplot(fig, width=None)


def main():
    # Wide mode
    st.set_page_config(layout="wide")
    # Designing the interface
    st.title("Siamese Classification")
    # For newline
    st.write("\n")

    cols = st.columns((1, 1))
    cols[0].header("Left Image")
    cols[1].header("Right Image")

    left_image, right_image, left_image_loc, right_image_loc = img_upload_side_bar(cols)
    negative_logit, positive_logit, cam_map = button(left_image, right_image)

    if negative_logit is not None and positive_logit is not None:
        prob = prediction(negative_logit, positive_logit)[
            negative_logit < positive_logit
        ]
        text = ["Different", "Similar"][negative_logit < positive_logit]
        st.subheader(f"{text}: {prob:.2%}")

    if cam_map is not None and len(cam_map) > 0:
        length = len(cam_map)
        cam_map_left, cam_map_right = cam_map[: length // 2], cam_map[length // 2 :]
        cam_vis(cam_map_left, left_image_loc, left_image)
        cam_vis(cam_map_right, right_image_loc, right_image)


if __name__ == "__main__":
    main()
