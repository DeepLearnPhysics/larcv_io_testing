import sys, os
import argparse
import time

from larcv import threadloader, queueloader
from larcv import distributed_queue_interface, distributed_larcv_interface

from mpi4py import MPI
COMM = MPI.COMM_WORLD

from src import larcv_io

import tempfile

from collections import OrderedDict

import json

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

    parser = argparse.ArgumentParser()


    subparsers = parser.add_subparsers(title='Datasets', 
                                      description='Available datasets', 
                                      dest='dataset', 
                                      help='Available datasets: sbnd, next')
  
    sbnd_parser = subparsers.add_parser('sbnd', help='SBND Cosmic Tagging Dataset')

    sbnd_parser.add_argument('--data-storage', type=str, choices=['sparse', 'dense'],
        help='Read data in sparse or dense format (different files)', required=True)
    
    sbnd_parser.add_argument('--data-output',  type=str,  choices=['sparse', 'dense'],
        help='Return data in sparse or dense format')

    sbnd_parser.add_argument('--resolution', type=str, choices=['fullres', 'downsample'],
                    help='Use the downsampled files or fullres files.', default='fullres')

    sbnd_parser.add_argument('--stream', type=str, choices=['train', 'test', 'dev'], default='test')
    
    next_parser = subparsers.add_parser('next', help='NEXT Cosmic Tagging Dataset')
    
    next_parser.add_argument('--data-output',  type=str,  choices=['sparse', 'dense'],
        help='Return data in sparse or dense format')

    next_parser.add_argument('--stream', type=str, choices=['train', 'test'], default='test')


    parser.add_argument('--io-mode', type=str, required=True,
                    choices=['queue', 'thread'],
                    help='Use Thread IO or Queue IO')


    parser.add_argument('--local-batch-size', type=int, required=True,
                    help='Number of images to read per local batch')

    parser.add_argument('--distributed', type=str2bool, required=True,
                    help='Run in distributed mode or not')

    parser.add_argument('--iterations', type=int, required=True,
                    help='Number of iterations to run')

    return parser






def gen_sparse2d_data_filler(name, producer, max_voxels):

    proc = larcv_io.ProcessConfig(proc_name=name, 
        proc_type='BatchFillerSparseTensor2D')

    proc.set_param('Verbosity',         '3')
    proc.set_param('Tensor2DProducer',  producer)
    proc.set_param('IncludeValues',     'true')
    proc.set_param('MaxVoxels',         max_voxels)
    proc.set_param('Channels',          '[0,1,2]')
    proc.set_param('UnfilledVoxelValue','-999')
    proc.set_param('Augment',           'false')

    return proc


def gen_dense2d_data_filler(name, producer, max_voxels):
    proc = larcv_io.ProcessConfig(proc_name=name, 
        proc_type='BatchFillerImage2D')

    proc.set_param('Verbosity',         '3')
    proc.set_param('ImageProducer',     producer)
    proc.set_param('Channels',          '[0,1,2]')

    return proc


def gen_sparse2d_to_dense2d_data_filler(name, producer, max_voxels):

    proc = larcv_io.ProcessConfig(proc_name=name, 
        proc_type='BatchFillerTensor2D')

    proc.set_param('Verbosity',         '3')
    proc.set_param('Tensor2DProducer',  producer)
    proc.set_param('IncludeValues',     'true')
    proc.set_param('Channels',          '[0,1,2]')
    proc.set_param('EmptyVoxelValue',   '0')

    return proc

def gen_sparse3d_data_filler(name, producer, max_voxels):

    proc = larcv_io.ProcessConfig(proc_name=name, 
        proc_type='BatchFillerSparseTensor3D')

    proc.set_param('Verbosity',         '3')
    proc.set_param('Tensor3DProducer',  producer)
    proc.set_param('IncludeValues',     'true')
    proc.set_param('MaxVoxels',         max_voxels)
    proc.set_param('UnfilledVoxelValue','-999')
    proc.set_param('Augment',           'true')

    return proc


def gen_label_filler(label_mode, prepend_names, n_classes):

    proc = larcv_io.ProcessConfig(proc_name=prepend_names + 'label', 
        proc_type='BatchFillerPIDLabel')

    proc.set_param('Verbosity',         '3')
    proc.set_param('ParticleProducer',  'label')
    proc.set_param('PdgClassList',      '[{}]'.format(','.join([str(i) for i in range(n_classes)])))

    return proc


def build_config_file(args):
    '''Using the provided args, build a config file
    
    
    Arguments:
        args {[type]} -- [description]
    '''

    if args.io_mode == 'queue':
        config = larcv_io.QueueIOConfig(name='IOTest')
    else:
        config = larcv_io.ThreadIOConfig(name='IOTest')


    file_name = get_file_name(args)

    # Unique process for each dataset:
    if args.dataset == 'sbnd':
        data_producer  = 'sbndwire'
        label_producer = 'sbnd_cosmicseg'
        if args.resolution == 'fullres':
            max_voxels = 80000
        else:
            max_voxels = 35000

        if args.data_storage == 'dense':
            data_proc = gen_dense2d_data_filler(name='data', 
                producer=data_producer, max_voxels=max_voxels)
            label_proc = gen_dense2d_data_filler(name='label', 
                producer=label_producer, max_voxels=max_voxels)
        elif args.data_storage == 'sparse':
            if args.data_output == 'dense':
                data_proc = gen_sparse2d_to_dense2d_data_filler(name='data', 
                    producer=data_producer, max_voxels=max_voxels)
                label_proc = gen_sparse2d_to_dense2d_data_filler(name='label', 
                    producer=label_producer, max_voxels=max_voxels)
            else:
                data_proc = gen_sparse2d_data_filler(name='data', 
                    producer=data_producer, max_voxels=max_voxels)
                label_proc = gen_sparse2d_data_filler(name='label', 
                    producer=label_producer, max_voxels=max_voxels)
    
    elif args.dataset == 'next':
        data_producer = ""
        label_producer = ""
        max_voxels = 20000
        raise Exception("NEXT data implementation not yet complete.")

    # Add the processes to the config:
    config.add_process(data_proc)
    config.add_process(label_proc)

    config.set_param('InputFiles', file_name)

    return config.generate_config_str()




    # This section of code determines the input file:


def get_file_name(args):
    with open('file_lookup.json', 'r') as f:
        datastore = json.load(f)

    if args.dataset == 'sbnd':
        datastore = datastore['sbnd']
        prefix = datastore['prefix']
        datastore = datastore[args.stream]

        datastore = datastore[args.resolution]
        datastore = datastore[args.data_storage]

        file_name = prefix + datastore

    elif args.dataset == 'next':
        datastore = datastore['next']
        prefix = datastore['prefix']
        datastore = datastore[args.stream]

        file_name = prefix + datastore
    else:
        raise Exception("Don't know what to do with this dataset: ", args.dataset)

    return file_name


def create_interface_object(args):

    config = build_config_file(args)



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


    # Generate a named temp file:
    main_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    main_file.write(config)

    main_file.close()

    # Prepare data managers:
    io_config = {
        'filler_name' : 'IOTest',
        'filler_cfg'  : main_file.name,
        'verbosity'   : 5,
        'make_copy'   : True
    }

    # By default, fetching data and label as the keywords from the file:
    data_keys = OrderedDict({
        'image': 'data', 
        'label': 'label'
        })

    if args.distributed:
        if args.io_mode == 'queue':
            larcv_interface.prepare_manager('primary', io_config, COMM.Get_size() * args.local_batch_size, data_keys, color=0)
        else:
            larcv_interface.prepare_manager('primary', io_config, COMM.Get_size() * args.local_batch_size, data_keys)
    else:
        if args.io_mode == 'queue':
            larcv_interface.prepare_manager('primary', io_config, args.local_batch_size, data_keys)
        else:
            larcv_interface.prepare_manager('primary', io_config, args.local_batch_size, data_keys)


    return larcv_interface

def event_loop(larcv_interface, args):

    loop_start = time.time()
    for i in range(args.iterations):
        iteration_start = time.time()
        if args.io_mode == 'queue':
            larcv_interface.prepare_next('primary')
            batch = larcv_interface.fetch_minibatch_data('primary')
            assert(len(batch) == args.local_batch_size)
        elif args.io_mode == 'thread':
            if not args.distributed:
                larcv_interface.next('primary')
            batch = larcv_interface.fetch_minibatch_data('primary')
            assert(len(batch) == args.local_batch_size)
        iteration_end = time.time()
        if args.verbose:
            print("Read local batch of {} in {:.2} s".format(args.local_batch_size, iteration_end - iteration_start))
    loop_end = time.time()

    if args.verbose:
        print("Time to read {} batches of global size {} in mode (distributed =={}): {:.3}".format(
                args.iterations,
                COMM.Get_size() * args.local_batch_size,
                args.distributed,
                loop_end - loop_start,
            )
        )

    return 


def main():

    parser = build_parser()

    args = parser.parse_args()

    # Add some arguments to the args:
    args.global_batch_size = COMM.Get_size() * args.local_batch_size
    args.verbose = False
    if COMM.Get_rank() == 0:
        args.verbose = True

    larcv_interface = create_interface_object(args)

    event_loop(larcv_interface, args)



if __name__ == '__main__':
    main()
