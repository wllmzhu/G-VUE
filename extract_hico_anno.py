import argparse
from preprocessing.hicodet.hico_constants import HicoConstants
from preprocessing.hicodet.mat_to_json import ConvertMat2Json


def extract_hico_anno(hico_dir, anno_dir):
    hico_const = HicoConstants(clean_dir=hico_dir, proc_dir=anno_dir)
    converter = ConvertMat2Json(hico_const)
    converter.convert()


def main():
    parser = argparse.ArgumentParser(description='specify two directories')
    parser.add_argument('--src', type=str, help='hico root directory')
    parser.add_argument('--dst', type=str, help='directory for json output')
    args = parser.parse_args()
    hico_dir = args.src
    anno_dir = args.dst
    extract_hico_anno(hico_dir=hico_dir, anno_dir=anno_dir)


if __name__ == '__main__':
    main()
