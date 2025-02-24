import argparse, os, torch
# from GAN import GAN
from WDNet import WDNet


"""parsing and configuration"""


def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--photo_path', type=str, default='dataset/photos', help='path to photos')
    parser.add_argument('--watermark_path', type=str, default='dataset/watermarks', help='path to watermark')
    parser.add_argument('--wallpaper_path', type=str, default='dataset/wallpapers', help='path to wallpaper')
    parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=16, help='The size of batch')
    # parser.add_argument('--input_size', type=int, default=28, help='The size of input image')
    parser.add_argument('--load_G_dir', type=str, default='models/WDNet_G_1.pth', help='Path to save the generator')
    parser.add_argument('--load_D_dir', type=str, default='models/WDNet_D_1.pth', help='Path to save the discriminator')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory name to save both model')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='log', help='Directory name to save training logs')
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--benchmark_mode', type=bool, default=True)
    parser.add_argument('--gpu', type=str, default='0')
    return check_args(parser.parse_args())


"""checking arguments"""


def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        os.makedirs(os.path.join(args.save_dir, "WDNet"))

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
        os.makedirs(os.path.join(args.result_dir, "WDNet", "result_vision"))
        os.makedirs(os.path.join(args.result_dir, "WDNet", "result_img"))

    # --log_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args


"""main"""


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    if args.benchmark_mode:
        torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    gan = WDNet(args)

    # launch the graph in a session
    gan.train()
    print(" [*] Training finished!")

    # visualize learned generator
    # gan.visualize_results(args.epoch)
    print(" [*] Testing finished!")


if __name__ == '__main__':
    main()
