import os
import time

import numpy

import logging
logger = logging.getLogger()

from larcv.config_builder import ConfigBuilder

class larcv_fetcher(object):

    def __init__(self, distributed, dataset, seed=0):


        random_access_mode = dataset.access_mode

        if distributed:
            from larcv import distributed_queue_interface
            self._larcv_interface = distributed_queue_interface.queue_interface(
                random_access_mode=random_access_mode.name, seed=seed)
        else:
            from larcv import queueloader
            self._larcv_interface = queueloader.queue_interface(
                random_access_mode=random_access_mode.name, seed=seed)

        self.distributed     = distributed
        self.dataset         = dataset

        self.writer     = None


    def __del__(self):
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.finalize()



    def prepare_sample(self, batch_size, color=0, start_index = 0):

        input_file = self.dataset.data_directory + "/" + self.dataset.file

        # First, verify the files exist:
        if not os.path.exists(input_file):
            raise Exception(f"File {input_file} not found")


        cb = ConfigBuilder()
        cb.set_parameter([input_file], "InputFiles")
        cb.set_parameter(5, "ProcessDriver", "IOManager", "Verbosity")
        cb.set_parameter(5, "ProcessDriver", "Verbosity")
        cb.set_parameter(5, "Verbosity")

        # Build up the data_keys:
        data_keys = {}
        data_keys['image'] = 'data'

        # Need to load up on data fillers.
        if self.dataset.dimension == 2:
            if self.dataset.input_shape.name == "sparse":
                cb.add_batch_filler(
                    datatype  = "sparse2d",
                    producer  = self.dataset.producer,
                    name      = "data",
                    MaxVoxels = 20000,
                    Augment   = False,
                    Channels  = list(self.dataset.channels)
                )
            elif self.dataset.output_shape.name == "dense":
                cb.add_batch_filler(
                    datatype  = "tensor2d",
                    producer  = self.dataset.producer,
                    name      = "data",
                    TensorType= "sparse",
                    Augment   = False,
                    Channels  = list(self.dataset.channels)
                )
        else:
            if self.dataset.input_shape.name == "sparse":
                if self.dataset.output_shape.name == "sparse":
                    cb.add_batch_filler(
                        datatype  = "sparse3d",
                        producer  = self.dataset.producer,
                        name      = "data",
                        MaxVoxels = 30000,
                        Augment   = False
                    )
                elif self.dataset.output_shape.name == "dense":
                    cb.add_batch_filler(
                        datatype  = "tensor3d",
                        producer  = self.dataset.producer,
                        name      = "data",
                        TensorType= "sparse",
                        Augment   = False,
                        Channels  = (0)
                    )


        logger.info(cb.print_config())


        # Prepare data managers:
        io_config = {
            'filler_name' : "iotest",
            'filler_cfg'  : cb.get_config(),
            'verbosity'   : 5,
            'make_copy'   : False
        }

        # Assign the keywords here:
        self.keyword_label = []
        for key in data_keys.keys():
            if key != 'image':
                self.keyword_label.append(key)




        self._larcv_interface.prepare_manager("iotest", io_config, batch_size, data_keys, color=color)



        while self._larcv_interface.is_reading("iotest"):
            time.sleep(0.1)

        return self._larcv_interface.size("iotest")




    def fetch_next_batch(self, name, force_pop=False):

        metadata=True

        pop = True
        if not force_pop:
            pop = False



        minibatch_data = self._larcv_interface.fetch_minibatch_data(name,
            pop=pop,fetch_meta_data=metadata)
        minibatch_dims = self._larcv_interface.fetch_minibatch_dims(name)




        # This brings up the next data to current data
        if pop:
            # print(f"Preparing next {name}")
            self._larcv_interface.prepare_next(name)
            # time.sleep(0.1)


        # If the returned data is None, return none and don't load more:
        if minibatch_data is None:
            return minibatch_data

        # Reshape as needed from larcv:
        for key in minibatch_data:
            if key == 'entries' or key == 'event_ids':
                continue
            minibatch_data[key] = numpy.reshape(minibatch_data[key], minibatch_dims[key])

        return minibatch_data
