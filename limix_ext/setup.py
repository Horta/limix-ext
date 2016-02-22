def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('limix_ext', parent_package, top_path)
    config.add_subpackage('gcta')
    config.add_subpackage('lmm')
    config.add_subpackage('ltmlm')
    config.add_subpackage('leap_')
    return config

if __name__ == '__main__':
    print('This is the wrong setup.py file to run')
