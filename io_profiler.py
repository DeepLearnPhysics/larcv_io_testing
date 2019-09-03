import sys, os
import argparse

from larcv import threadloader, queueloader
from larcv import distributed_queue_interface, distributed_larcv_interface


# This function is to parse strings from argparse into bool
def str2bool(v):
    '''Convert string to boolean value
    
    This function is from stackoverflow: 
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    
    Arguments:
        v {str} -- [description]
    
    Returns:
        bool -- [description]
    
    Raises:
        argparse -- [description]
    '''
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def build_parser():

    parser = arparse.ArgumentParser()


    subparsers = parser.add_subparsers(title='Datasets', 
                                      description='Available datasets', 
                                      dest='dataset', 
                                      help='Available datasets: sbnd, next')
  
    sbnd_parser = subparsers.add_parser('sbnd', help='SBND Cosmic Tagging Dataset')

    sbnd_parser.add_argument('--data-storage', type=str, choices=['sparse', 'dense'],
        help="Read data in sparse or dense format (different files)")
    
    sbnd_parser.add_argument('--data-output',  type=str,  choices=['sparse', 'dense'],
        help="Return data in sparse or dense format")

    sbnd_parser.add_argument('--downsampled', type=str2bool,
                    help='Use the downsampled files.')
    
    next_parser = subparsers.add_parser('next', help='NEXT Cosmic Tagging Dataset')
    
    next_parser.add_argument('--data-output',  type=str,  choices=['sparse', 'dense'],
        help="Return data in sparse or dense format")



    parser.add_argument('--io-mode', type=str, choices=['queue', 'thread'],
                    help='Use Thread IO or Queue IO')


    parser.add_argument('--local-batch-size', type=int,
                    help='Number of images to read per local batch')

    parser.add_argument('--distributed', type=str2bool,
                    help='Run in distributed mode or not')

    parser.add_argument('--iterations', type=int,
                    help='Number of iterations to run')




def build_config_file(args):

    if args.distributed:
        if args.io_mode == 'queue':
            larcv_interface = distributed_queue_interface.queue_interface()
        else:
            larcv_interface = distributed_larcv_interface.thread_interface()
    else:
        if args.io_mode == 'queue':
            larcv_interface = queueloader.queue_interface()
        else:
            larcv_interface = threadloader.thread_interface()

    print(larcv_interface)







        # # This is a dummy placeholder, you must check this yourself:
        # if 640 in FLAGS.SHAPE:
        #     max_voxels = 35000
        # else:
        #     max_voxels = 70000

        # # Use the templates to generate a configuration string, which we store into a temporary file
        # if FLAGS.TRAINING:
        #     config = io_templates.train_io(
        #         input_file=FLAGS.FILE, 
        #         data_producer= FLAGS.IMAGE_PRODUCER,
        #         label_producer= FLAGS.LABEL_PRODUCER, 
        #         max_voxels=max_voxels)
        # else:
        #     config = io_templates.ana_io(
        #         input_file=FLAGS.FILE, 
        #         data_producer= FLAGS.IMAGE_PRODUCER,
        #         label_producer= FLAGS.LABEL_PRODUCER, 
        #         max_voxels=max_voxels)


        # # Generate a named temp file:
        # main_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        # main_file.write(config.generate_config_str())

        # main_file.close()
        # self._cleanup.append(main_file)

        # # Prepare data managers:
        # io_config = {
        #     'filler_name' : config._name,
        #     'filler_cfg'  : main_file.name,
        #     'verbosity'   : FLAGS.VERBOSITY,
        #     'make_copy'   : True
        # }

        # # By default, fetching data and label as the keywords from the file:
        # data_keys = OrderedDict({
        #     'image': 'data', 
        #     'label': 'label'
        #     })


        # self._larcv_interface.prepare_manager('primary', io_config, FLAGS.MINIBATCH_SIZE, data_keys, color)

def main():

    parser = build_parser()

    args = parser.parse_args()


    build_config_file(args)
