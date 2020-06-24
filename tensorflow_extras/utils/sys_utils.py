import psutil
import GPUtil
import tensorflow as tf

def get_sys_stats():
    '''
    Grab system information
    '''
    gpus = GPUtil.getGPUs()
    stats = {}
    stats['gpu_util'] = tf.reduce_mean([gpu.load for gpu in gpus])*100
    stats['gpu_mem'] = tf.reduce_mean([gpu.memoryUtil for gpu in gpus])*100
    stats['cpu_util'] = tf.convert_to_tensor(psutil.cpu_percent())
    stats['sys_mem'] = tf.convert_to_tensor(psutil.virtual_memory().percent)
    return stats