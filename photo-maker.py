# -*- coding:utf-8 -*-

"""
用于生成各种有趣图片的简单脚本
"""

import os
import argparse
import random
import textwrap
import itertools
import multiprocessing
from datetime import datetime
from functools import partial
from PIL import Image, ImageOps, ImageStat, PngImagePlugin

# Python 2 && Python3
zip = getattr(itertools, 'izip', zip)

__author__ = 'shellvon'
__email__ = 'iamshellvon@gmail.com'

# Dirty Fixed shellvon/photo-maker#1
# See https://github.com/python-pillow/Pillow/issues/1434

PngImagePlugin.iTXt.__getnewargs__ = lambda x: (x.__str__(), '', '')

cli = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                              description=textwrap.dedent('''\
                                                        A Dead Simple Photo Maker
                                            ===========================================================
                                            ______ _           _             ___  ___      _                       
                                            | ___ \ |         | |            |  \/  |     | |                      
                                            | |_/ / |__   ___ | |_ ___ ______| .  . | __ _| | _____ _ __           
                                            |  __/| '_ \ / _ \| __/ _ \______| |\/| |/ _` | |/ / _ \ '__|          
                                            | |   | | | | (_) | || (_) |     | |  | | (_| |   <  __/ |             
                                            \_|   |_| |_|\___/ \__\___/      \_|  |_/\__,_|_|\_\___|_|  

                                            The most commonly used photo-maker commands are:
                                                heart      Use multi image files to generate a heart picture
                                                split      Split one image into A x B multi grid pictures
                                                mosaic     Generate a mosaic photo by use tiles
                                                ascii      Convert an image to ascii text 
                                            '''),
                              epilog='Source Code: http://github.com/shellvon/photo-maker')
subparsers = cli.add_subparsers(dest='subcommand')

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(30, 38)

IS_WIN = os.name == 'nt'


def echo(msg, level='INFO'):
    """支持颜色输出"""
    level = level.upper()

    def colored(text, color=30, bgcolor=49, style=0):
        """
        color   30:黑,31:红,32:绿,33:黄,34:蓝色,35:紫色,36:深绿,37:白色
        bgcolor 40:黑, 41:深红,42:绿,43:黄色,44:蓝色,45:紫色,46:深绿,47:白色
        style   0:无样式,1:高亮/加粗,4:下换线,7:反显
        """
        hehe = {'text': text, 'color': color, 'bgcolor': bgcolor, 'style': style}
        str_to_color = "\033[{style};{color};{bgcolor}m{text}\033[0m".format(**hehe)
        return str_to_color

    level_to_color_map = {
        'WARNING': YELLOW,
        'INFO': WHITE,
        'DEBUG': BLUE,
        'ERROR': RED,
    }
    if not IS_WIN and level in level_to_color_map:
        level_color = level_to_color_map[level]
        bold = 1
    else:
        bold = 0
        level_color = BLACK
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    prefix = colored(
        '[{level_name:<8s}{time_str}]'.format(level_name=level,
                                              time_str=now),
        color=level_color, style=bold)

    msg = colored(msg, color=level_color, bgcolor=49, style=0)
    print('{prefix} {msg}'.format(prefix=prefix, msg=msg))


def arguments(*args, **kwargs):
    return args, kwargs


def subcommand(args=None, parent=subparsers):
    """ Copied from https://gist.github.com/mivade/384c2c41c3a29c637cb6c603d4197f9f#file-cli-py-L15-L39 """

    args = args or []

    def decorator(func):
        parser = parent.add_parser(func.__name__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                   description=textwrap.dedent(func.__doc__))
        for arg in args:
            parser.add_argument(*arg[0], **arg[1])
        parser.set_defaults(func=func)

    return decorator


def size_type(s, sep='x'):
    try:
        x, y = map(int, s.split(sep))
        return x, y
    except:
        raise argparse.ArgumentTypeError('the argument format must be A%sB' % sep)


@subcommand([
    arguments('-d', '--dir', dest='dir', help='directory of the images, default is current dir', default='.'),
    arguments('--autofill', dest='autofill', help='autofill pictures when picture is too few, default is False',
              action='store_true', default=False),
    arguments('--padding', dest='padding', help='padding size between each grid, default is 10',
              default=10, type=int),
    arguments('--size', dest='size', help='output image size，format is AxB, default size is: 1920x1920',
              default=(1920, 1920), type=size_type),
    arguments('-s', '--shuffle', dest='shuffle', help='shuffle the image file list', action='store_true',
              default=False),
    arguments('-e', '--extensions', dest='extensions', help='image file type, default is png,jpg,jpeg',
              default=('png', 'jpg', 'jpeg'),
              nargs='+'),
    arguments('--intermediate', dest='intermediate', help='save intermediate file, default is False',
              action='store_true',
              default=False),
    arguments('-o', '--output', dest='output', help='output image filename, default is collage.png',
              default='collage.png'),

])
def heart(args):
    """
    微信心型九空格图片,一共是分享的9张,每一张也是9格图
    如下图，需要填满的位置是X所在的位置:

    Fig 1:              Fig 2:         Fig 3:
    +---+---+---+   +---+---+---+   +---+---+---+
    | 1 | 2 | 3 |   | 1 | 2 | 3 |   | 1 | 2 | 3 |
    +-----------+   +-----------+   +-----------+
    | 4 | 5 | x |   | x | 5 | x |   | x | 5 | 6 |
    +-----------+   +-----------+   +-----------+
    | 7 | x | x |   | x | x | x |   | x | x | 9 |
    +-----------+   +-----------+   +-----------+

    Fig 4:              Fig 5:         Fig 6:
    +---+---+---+   +---+---+---+   +---+---+---+
    | x | x | x |   | x | x | x |   | x | x | x |
    +-----------+   +-----------+   +-----------+
    | x | x | x |   | x | x | x |   | x | x | x |
    +-----------+   +-----------+   +-----------+
    | 7 | x | x |   | x | x | x |   | x | x | 9 |
    +-----------+   +-----------+   +-----------+

    Fig 7:              Fig 8:         Fig 9:
    +---+---+---+   +---+---+---+   +---+---+---+
    | 1 | 2 | x |   | x | x | x |   | x | 2 | 3 |
    +-----------+   +-----------+   +-----------+
    | 4 | 5 | 6 |   | x | x | x |   | 4 | 5 | 6 |
    +-----------+   +-----------+   +-----------+
    | 7 | 8 | 9 |   | 7 | x | 9 |   | 7 | 8 | 9 |
    +-----------+   +-----------+   +-----------+
                 ^^^
                  |
                  +----- 每一张图的间隔称之为 Padding

    因此一共需要的图片张数是: (3 * 2) + 5 + (8 * 2) + 9 + (1 * 2) + 7 = 45 张图.
    """
    image_files = [os.path.join(args.dir, fn) for fn in os.listdir(args.dir) if
                   fn.split('.')[-1].lower() in args.extensions]
    if not image_files:
        echo('Bad directory! Please select other directory containing images', 'error')
        exit(1)
    echo('Got [%d] images to make photo collage with extension: %s' % (len(image_files), args.extensions))
    if args.shuffle:
        echo('Shuffle the images')
        random.shuffle(image_files)

    size = args.size
    padding = args.padding
    save_intermediate = args.intermediate
    if size[0] != size[1]:
        echo('Size error! Heart layout picture must be square.', 'error')
        exit(1)

    l = len(image_files)
    pics_cnt_table = [3, 5, 3, 8, 9, 8, 1, 7, 1]
    total_pic_needed = sum(pics_cnt_table)
    if l < total_pic_needed and not args.autofill:
        echo('Too few pictures to generate heart layout, need %d pictures.' % (total_pic_needed,), 'error')
        exit(1)
    # auto fill
    while l < total_pic_needed:
        image_files.extend(image_files)
        l = len(image_files)

    grid_size = tuple(map(lambda x: (x - padding * 2) // 3, size))  # 9张小图每一张图的大小
    cell_w, cell_h = grid_size[0] // 3, grid_size[1] // 3  # 小图中每一张图的大小

    # 将图绘制到canvas中去
    def draw_to_canvas(canvas, imgs, left=True):
        boxes = {
            1: [(cell_w * 2 if left else 0, 0)],  # 左/右下角

            3: [(cell_w * 2 if left else 0, cell_h),  # 左/右上角
                (cell_w, cell_h * 2),
                (cell_w * 2 if left else 0, cell_h * 2),
                ],
            5: [(0, cell_h), (2 * cell_w, cell_h), (0, cell_h * 2), (cell_w, cell_h * 2), (cell_w * 2, cell_h * 2)],
            # 正上方

            8: [(0, 0), (cell_w, 0), (cell_w * 2, 0), (0, cell_h), (cell_w, cell_h), (cell_w * 2, cell_h),
                (cell_w, cell_h * 2), (cell_w * 2 if left else 0, cell_h * 2)],  # 左右方

            9: [(0, 0), (cell_w, 0), (cell_w * 2, 0), (0, cell_h), (cell_w, cell_h), (cell_w * 2, cell_h),
                (0, cell_h * 2), (cell_w, cell_h * 2), (cell_w * 2, cell_h * 2)],  # 中间

            7: [(0, 0), (cell_w, 0), (cell_w * 2, 0), (0, cell_h), (cell_w, cell_h), (cell_w * 2, cell_h),
                (cell_w, cell_h * 2)],  # 正下方
        }
        l = len(imgs)
        for idx, fn in enumerate(imgs):
            im = ImageOps.fit(Image.open(fn), size=(cell_w, cell_h), method=Image.ANTIALIAS)
            canvas.paste(im, box=boxes[l][idx], mask=im.convert('RGBA'))

    background = Image.new('RGBA', size=size, color=(0, 0, 0, 0))
    for seq in range(9):
        canvas = Image.new('RGBA', size=grid_size, color=(255, 255, 255, 0))
        need_pic_cnt = pics_cnt_table[seq]
        sources, images = image_files[:need_pic_cnt], image_files[need_pic_cnt:]
        left = (seq % 3) == 0
        echo('Draw images at seq %d, is left: %s, cnt: %d, save intermediate: %s' % (
            seq, left, need_pic_cnt, save_intermediate))
        draw_to_canvas(canvas, sources, left=left)
        box = ((seq % 3) * (grid_size[0] + padding), (seq // 3) * (grid_size[1] + padding))
        background.paste(canvas, box=box)
        if save_intermediate:
            canvas.save('intermediate_%d.png' % seq)
    background.save(args.output)
    echo('All done, Save result file to : %s' % args.output)


@subcommand([
    arguments('-i', '--input', dest='input', help='the file to split', required=True),
    arguments('--grid', dest='grid', help='grid size,  default is 3x3', default=(3, 3), type=size_type),
    arguments('--prefix', dest='prefix', help='filename prefix, default is grid_', default='grid_'),
    arguments('-o', '--output', dest='output', help='output dir name, default is current dir', default='.'),
])
def split(args):
    """将图拆分成 AxB 的多张小网格图"""
    try:
        source_im = read_img(args.input, mode='RGB')
    except IOError:
        echo('Cannot open the source file: %s' % args.input, 'error')
        return
    width, height = source_im.size
    echo('Image size is:  %d x %d ' % (width, height), 'debug')
    result = split_img(source_im, args.grid)
    format = source_im.format.lower()
    for idx, im in enumerate(result):
        im.save('%s/%s%d.%s' % (args.output, args.prefix, idx, format))
    echo('All done, Save result file to dir: %s' % args.output)


@subcommand([
    arguments('-i', '--input', dest='input', help='the source image to generate', required=True),
    arguments('-r', '--ratio', dest='ratio', help='the ratio between the row/col and src image size, default is 32',
              default=32, type=int),
    arguments('--no-repeat', dest='no_repeat', help='no repeat use tile', action='store_true', default=False),
    arguments('-o', '--output', dest='output', help='the output filename, default is: mosaic.png',
              default='mosaic.png'),
    arguments('-d', '--dir', dest='dir', help='directory of the images, default is ./tiles/emoji',
              default='./tiles/emoji'),
    arguments('-e', '--extensions', dest='extensions', help='tile image file type, default is png,jpg,jpeg',
              default=('png', 'jpg', 'jpeg'),
              nargs='+'),
    arguments('-z', '--zoom', dest='zoom', help='zoom zhe result image, default is 1', default=1, type=int),
])
def mosaic(args):
    """
    相片马赛克, 即蒙太奇照片。 给定一张照片,和一个文件夹目录,系统将利用文件夹内
    所对应的照片生成给定的这张照片,为了达到这个效果,文件夹内的目录的照片越多越好

    Step 0: 预处理tiles文件(处理成相同大小/长宽比等)
    Step 1: 读取目标文件(input source file) 并将其拆分为 AxB 的网格.
    Step 2: 对于每一个网格图块，寻找与源文件最佳匹配的图。
    Step 3: 粘贴 保存


     Fig 1: 将源图中每一个小格图放大到右边(原图中被拆分为3 X 3)
       源文件            (i*w, i*j)
    +---+---+---+       +-----------+
    | 1 | 2 | 3 | ----> |           |
    +-----------+       +           + 高度为h
    | 4 | 5 | 6 |       |           |
    +-----------+       +-----------+  (i+1)*w, (j+1)*h
    | 7 | 8 | 9 |           宽度为w
    +-----------+

      一张图都可以用8bit的RGB颜色来表示,给定一张图包含N像素,那么他的平均RGB计算如下:
            (R,G,B)avg = (Σ(Ri..n)/N,  Σ(Gi..n)/N,  Σ(Bi..n)/N)
      为了寻找两张图最相近的方式之一就是通过比较彼此的平均RGB相差多远. 相差的距离计算公式:
            Distance(r,g,b) = sqrt((r1-r2)^2 + (g1-g2)^2 + (b1-b2)^2)
      距离最小者则最相似.
    """

    try:
        source_im = read_img(args.input, mode='RGB')
    except IOError:
        echo('Cannot open the source file: %s' % args.input, 'error')
        return
    original_width, original_height = source_im.size
    if args.ratio <= 0:
        echo('Ratio must be >= 1', 'error')
        return
    if not os.path.exists(args.dir):
        echo('Dir [%s] does not exists' % args.dir, 'error')
        return

    zoom = args.zoom

    row, col = original_height // args.ratio, original_width // args.ratio

    echo('Split the input images, source file size: %s, zoom: %d, grid size: %s' % (
        source_im.size, args.zoom, (row, col)))
    grid_images = split_img(source_im, (row, col))

    tile_size = zoom * original_width // col, zoom * original_height // row

    worker_cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(worker_cnt)

    filenames = list(get_tile_filenames(args.dir, args.extensions))
    tiles_count = len(filenames)
    if tiles_count < 50:
        echo('Tiles not enough in dir %s with extensions %s (total found: %d)' % (
            args.dir, args.extensions, tiles_count), 'warning')
        return
    chunk_size = min(100, len(filenames))
    echo('Got %d tile files, tile size: %s' % (len(filenames), tile_size))

    echo('Prepare tiles')

    # 获取tiles Images对象
    tiles = list(flatten(pool.map(partial(load_tiles, size=tile_size), list(group_chunk(filenames, chunk_size)))))
    echo('Calculate average color of tiles')
    # 批量获取每一张tile图片的平均RGB值
    avg_color_of_tile_imgs = pool.map(get_average_color, tiles)

    echo('Calculate average color of source image')
    # 批量获取每一张源文件grid图的RGB值
    avg_color_of_source_imgs = pool.map(get_average_color, grid_images)

    background = Image.new('RGB', (tile_size[0] * col, tile_size[1] * row))

    echo('Try to generating mosaic pic')
    for seq, color in enumerate(avg_color_of_source_imgs):
        closet_match = pool.map(partial(find_closet_match, color),
                                list(group_chunk(avg_color_of_tile_imgs, chunk_size)))
        best_matched = min(((idx, el) for idx, el in enumerate(closet_match)), key=lambda x: x[1][1])
        chunk_index, (index, diff) = best_matched
        best_index = chunk_index * chunk_size + index
        best_tile = tiles[best_index]
        m = seq % col  # 当前tile 所在的二维坐标(列)
        n = seq // col  # 当前tile 所在的二维坐标(行)

        if seq % 300 == 0:
            echo('Get best tile at pos: (%d, %d) file: %s, diff: %.2f' % (n, m, filenames[best_index], diff), 'debug')
        background.paste(best_tile, (tile_size[0] * m, tile_size[1] * n))
    pool.close()
    pool.join()
    # background.show()
    echo('Save result file to: %s' % args.output)
    background.save(args.output)
    echo('All done')


def get_tile_filenames(directory, extensions=None):
    echo('Lookup the tile filename list from %s' % directory, 'debug')
    extensions = extensions or ['jpg', 'png', 'jpeg']
    for fn in os.listdir(directory):
        ext = fn.split('.')[-1].lower()
        if ext not in extensions:
            continue
        fn = os.path.join(directory, fn)
        yield fn


def load_tiles(filenames, size=(32, 32)):
    echo('Try to load tiles from %d files, size: %s' % (len(filenames), size), 'debug')
    tiles = [ImageOps.fit(read_img(fn, mode='RGB'), size, method=Image.ANTIALIAS) for fn in filenames]
    echo('Got %d tiles' % len(tiles), 'debug')
    return tiles


def get_average_color(im):
    """获取图片颜色平均值"""
    return ImageStat.Stat(im).mean


def split_img(image, grid_size):
    """拆图"""
    width, height = image.size[0], image.size[1]
    m, n = grid_size
    w, h = width // n, height // m
    return [image.crop((i * w, j * h, (i + 1) * w, (j + 1) * h)) for j in range(m)
            for i in range(n)]


def find_closet_match(color, tile_colors):
    """在tile_colors中寻找与颜色color最接近的颜色,返回下标和差异值"""
    diff = float('inf')
    index = 0
    for i, c in enumerate(tile_colors):
        d = get_color_diff(color, c)
        if d < diff:
            diff = d
            index = i
    return index, diff


def get_color_diff(color1, color2):
    """计算俩颜色之间的差异"""
    # 小优化, 因为 sqrt 函数是单调递增的,可以不用开方..
    return (color1[0] - color2[0]) ** 2 + (color1[1] - color2[1]) ** 2 + (color1[2] - color2[2]) ** 2


def group_chunk(iterable, chunk_size=20):
    """将一个迭代器拆分成多个chunk"""
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i:i + chunk_size]


def flatten(iterable):
    """把一个嵌套的结构展开"""
    return itertools.chain.from_iterable(iterable)


@subcommand([
    arguments('-i', '--input', dest='input', help='the file to generate ascii', required=True),
    arguments('--col', dest='col', help='the output column, default is 120', type=int, default=120),
    arguments('-s', '--simple', dest='simple', help='use simple output mode (less ascii char) default is false',
              action='store_true',
              default=False),
    arguments('-o', '--output', dest='output', help='output filename, default is <STDOUT>', type=argparse.FileType('w'),
              default='-'),

])
def ascii(args):
    """
    将给定的照片转化为 ASCII 码展示,这种技术叫做 Ascii Art
    参见: https://www.wikiwand.com/en/ASCII_art

    思路:
        Step 1. 把源文件拆分成多个小图(A X B 形式的小图,由参数col指定)
        Step 2. 计算每一张小图的灰度强弱(calculate intensity, A: 再次利用平均RGB, B: 转化为greyscale图.)
        Step 3. 替换为对应强度的字符进行展示
    """
    # Please See here: http://paulbourke.net/dataformats/asciiart/
    grey_scale = '''$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,"^`'. '''
    if args.simple:
        echo('Use simple mode..', 'debug')
        grey_scale = '@%#*+=-:. '
    try:
        source_im = read_img(args.input)
    except IOError:
        echo('Cannot open the source file: %s' % args.input, 'error')
        return
    original_width, original_height = source_im.size
    echo('Got image size: %d x %d ' % (original_width, original_height))

    # 计算需要的行和列
    col = min(args.col, original_width)
    row = int(original_height * col * 0.5 / original_width)
    tile_imgs = split_img(source_im, (row, col))
    pool = multiprocessing.Pool()
    echo('Try to calculate intensity')

    ascii_image_chars = pool.map(partial(get_string_by_imgs, chars_table=grey_scale),
                                 group_chunk(tile_imgs, col))
    pool.close()
    pool.join()

    echo('Write to file: %s' % args.output.name)
    args.output.write('\n'.join(ascii_image_chars))
    echo('\n\nAll done!')


def get_string_by_imgs(img, chars_table='@%#*+=-:. '):
    """通过图片获取对应需要展示的字符"""
    size = len(chars_table) - 1
    chars = []
    for im in img:
        avg_color = get_average_color(im)
        intensity = int(size * sum(avg_color) * 1. / len(avg_color) / 255)
        chars.append(chars_table[intensity])
    return ''.join(chars)


def read_img(file_or_image, fixed_orientation=True, mode=None):
    """读取图片文件，但是支持修复方向
    See https://github.com/shellvon/photo-maker/issues/2
    """
    if isinstance(file_or_image, (Image.Image,)):
        im = input
    else:
        im = Image.open(file_or_image)
    if not fixed_orientation:
        return im.convert(mode) if mode else im

    # 不能先convert 否则exif信息读不到.
    exif_orientation_tag = 274
    orientation_maps = {
        3: Image.ROTATE_180,
        6: Image.ROTATE_270,
        8: Image.ROTATE_90,
    }
    try:
        orientation = im._getexif()[exif_orientation_tag]
        if orientation in orientation_maps:
            degree = orientation_maps[orientation]
            # use rotate with expand param is same as transpose
            # See https://github.com/python-pillow/Pillow/issues/1746
            im = im.transpose(degree)
    except (TypeError, AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        pass
    if mode:
        im = im.convert(mode)
    return im


def main():
    args = cli.parse_args()
    if args.subcommand is None:
        cli.print_help()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
