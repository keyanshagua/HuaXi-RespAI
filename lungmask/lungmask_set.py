import argparse
import os
import sys
from importlib import metadata
import ants
import numpy as np
import SimpleITK as sitk
from lungmask import LMInferer, utils
from lungmask.logger import logger

def path(string):
    if os.path.exists(string):
        return string
    else:
        sys.exit(f"File not found: {string}")

def apply_mask_and_crop(input_image_path, input_mask_path, output_cropped_path):
    # 加载原始图像和 mask
    image = ants.image_read(input_image_path)
    mask = ants.image_read(input_mask_path)

    # 应用 mask 到原始图像
    masked_image = image * mask

    # 裁剪图像
    cropped_image = ants.crop_image(masked_image, mask)

    # 保存裁剪后的图像
    ants.image_write(cropped_image, output_cropped_path)

def process_file(input_image_path, output_mask_path, output_cropped_path, args):
    logger.info(f"Processing {input_image_path}")

    # 使用 SimpleITK 直接加载 .nii.gz 图像
    input_image = sitk.ReadImage(input_image_path)

    logger.info("Infer lungmask")
    inferer = LMInferer(
        modelname=args.modelname,
        modelpath=args.modelpath,
        force_cpu=args.cpu,
        batch_size=args.batchsize,
        volume_postprocessing=not (args.nopostprocess),
        tqdm_disable=args.noprogress,
    )
    result = inferer.apply(sitk.GetArrayFromImage(input_image))

    # 将所有非零值设置为1，统一标记左右肺部
    result[result > 0] = 1

    result_out = sitk.GetImageFromArray(result)
    result_out.CopyInformation(input_image)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(output_mask_path)

    if not args.removemetadata:
        writer.SetKeepOriginalImageUID(True)
        DICOM_tags_to_keep = utils.get_DICOM_tags_to_keep()
        for key in input_image.GetMetaDataKeys():
            if key in DICOM_tags_to_keep:
                result_out.SetMetaData(key, input_image.GetMetaData(key))
        result_out.SetMetaData("0008|103e", "Created with lungmask")
        result_out.SetMetaData("0028|1050", "1")  # Window Center
        result_out.SetMetaData("0028|1051", "2")  # Window Width

    logger.info(f"Save lungmask to: {output_mask_path}")
    writer.Execute(result_out)

    # 调用 apply_mask_and_crop 函数
    apply_mask_and_crop(input_image_path, output_mask_path, output_cropped_path)

    logger.info(f"Save cropped image to: {output_cropped_path}")

def main():
    version = metadata.version("lungmask")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input",
        metavar="input",
        type=path,
        help="Path to the input image or directory containing images",
    )
    parser.add_argument(
        "output",
        metavar="output",
        type=str,
        help="Directory for output lungmask and cropped images",
    )
    parser.add_argument(
        "--modelname",
        help="specifies the trained model, Default: R231",
        type=str,
        choices=["R231", "LTRCLobes", "LTRCLobes_R231", "R231CovidWeb"],
        default="R231",
    )
    parser.add_argument(
        "--modelpath", help="specifies the path to the trained model", default=None
    )
    parser.add_argument(
        "--cpu",
        help="Force using the CPU even when a GPU is available, will override batchsize to 1",
        action="store_true",
    )
    parser.add_argument(
        "--nopostprocess",
        help="Deactivates postprocessing (removal of unconnected components and hole filling)",
        action="store_true",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        help="Number of slices processed simultaneously. Lower number requires less memory but may be slower.",
        default=20,
    )
    parser.add_argument(
        "--noprogress",
        action="store_true",
        help="If set, no tqdm progress bar will be shown",
    )
    parser.add_argument(
        "--version",
        help="Shows the current version of lungmask",
        action="version",
        version=version,
    )
    parser.add_argument(
        "--removemetadata",
        action="store_true",
        help="Do not keep study/patient related metadata of the input, if any. Only affects output file formats that can store such information (e.g. DICOM).",
    )

    args = parser.parse_args()

    # 检查输入路径是目录还是文件
    if os.path.isdir(args.input):
        # 遍历目录中的所有 .nii.gz 文件
        for file_name in os.listdir(args.input):
            if file_name.endswith(".nii.gz"):
                input_image_path = os.path.join(args.input, file_name)
                output_mask_path = os.path.join(args.output, file_name.replace(".nii.gz", "_mask.nii.gz"))
                output_cropped_path = os.path.join(args.output, file_name.replace(".nii.gz", "_cropped.nii.gz"))
                process_file(input_image_path, output_mask_path, output_cropped_path, args)
    else:
        # 单个文件处理
        input_image_path = args.input
        file_name = os.path.basename(input_image_path)
        output_mask_path = os.path.join(args.output, file_name.replace(".nii.gz", "_mask.nii.gz"))
        output_cropped_path = os.path.join(args.output, file_name.replace(".nii.gz", "_cropped.nii.gz"))
        process_file(input_image_path, output_mask_path, output_cropped_path, args)


if __name__ == "__main__":
    print("called as script")
    main()
