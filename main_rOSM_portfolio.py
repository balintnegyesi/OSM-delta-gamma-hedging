"""
The main file to run BSDE solver to solve parabolic partial differential equations (PDEs).

"""

import json
import munch
import os
import logging
import csv

from absl import app
from absl import flags
from absl import logging as absl_logging
import numpy as np
import tensorflow as tf
from datetime import datetime
import warnings

# import equation as eqn
from lib.equations import portfolios as eqn
from lib.OSM.systemOSM import systemOSM as BSDESolver
flags.DEFINE_string('config_path', 'configs/',  # eqn_name to be appended here
                        """The path to load json file.""")  # original: hjb_lq_d100.json
flags.DEFINE_string('filename', None, """The json file under ./configs/ containing the relevant setup for this EQ.""")
flags.DEFINE_string('exp_name', None,
                        """The name of numerical experiments, prefix for logging""")  # can be overwritten by the user
FLAGS = flags.FLAGS


def main(argv):
    # tf.config.run_functions_eagerly(True)
    # physical_devices = tf.config.list_physical_devices('GPU')
    # gpus = tf.config.experimental.list_physical_devices('GPU')

    # try:
    #     # Disable all GPUS
    #     tf.config.set_visible_devices([], 'GPU')
    #     visible_devices = tf.config.get_visible_devices()
    #     for device in visible_devices:
    #         assert device.device_type != 'GPU'
    # except:
    #     # Invalid device or cannot modify virtual devices once initialized.
    #     pass
    #
    # try:
    #     logging.info("\n\n\n\nPENIS")
    #     for gpu in gpus:
    #         tf.config.experimental.set_memory_growth(gpu, True)
    #     # tf.config.gpu.set_per_process_memory_growth(True)
    #     logging.info("@DO SOMETHING WITH ME FOR FUCK'S SAKE\n\n\n\n\n\n\n\n\n\n\n\n")
    #     logging.info("@GPU: set_memory_growth=True")
    # except:
    #     # Invalid device or cannot modify virtual devices once initialized.
    #     pass

    try:
        FLAGS.filename = argv[1]  # filenames
        FLAGS.config_path = FLAGS.config_path + FLAGS.filename
    except IndexError:
        raise FileNotFoundError('Please provide an input JSON configuration file under "./configs"')
    with open(FLAGS.config_path) as json_data_file:
        config = json.load(json_data_file)
    config = munch.munchify(config)
    tf.keras.backend.set_floatx(config.net_config.dtype)
    try:
        tf.config.run_functions_eagerly(config.net_config.is_debugging)
    except:
        pass
    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)

    today = datetime.today().strftime('%d_%m_%Y')
    equation = config.eqn_config.eqn_name
    try:
        FLAGS.exp_name = equation + '_' + argv[2] + '_' + today  # directory to save logs under
        FLAGS.exp_name = equation + '_' + argv[2]  # directory to save logs under
    except IndexError:
        warnings.warn('No experiment name provided: will use the default one [filename_d_m_Y]')
        FLAGS.exp_name = equation + '_' + today
        FLAGS.exp_name = equation
    if config.eqn_config.discretization_config.theta_y == 0.5:
        FLAGS.log_dir = './logs/systemOSM_theta0p5/' + FLAGS.exp_name
    elif config.eqn_config.discretization_config.theta_y == 1:
        FLAGS.log_dir = './logs/systemOSM_theta1/' + FLAGS.exp_name
    elif config.eqn_config.discretization_config.theta_y == 0:
        FLAGS.log_dir = './logs/systemOSM_theta0/' + FLAGS.exp_name
    else:
        raise NotImplementedError

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    with open(FLAGS.log_dir + '/config.json', 'w') as outfile:
        json.dump(dict((name, getattr(config, name))
                       for name in dir(config) if not name.startswith('__')),
                  outfile, indent=2)

    absl_logging.get_absl_handler().setFormatter(logging.Formatter('%(levelname)-6s %(message)s'))
    absl_logging.set_verbosity('info')

    logging.info('Begin to solve %s ' % config.eqn_config.eqn_name)
    bsde_solver = BSDESolver(config, bsde)
    training_history = bsde_solver.train_fetch(output_dir=FLAGS.log_dir)

    if config.net_config.is_report_t0_errors:
        path_t0_sol = FLAGS.log_dir + "/solutions_at_t0"
        if not os.path.exists(path_t0_sol):
            os.makedirs(path_t0_sol)

        logging.info("=" * 50 + "\nComparison against analytical solution")
        x0 = bsde.sample(1)[1][:, :, 0]

        if not bsde_solver.is_nn_at_t0:
            y_est = bsde_solver.Y[0].numpy()[0, ...]
        else:
            y_est = bsde_solver.Y[0](x0, False).numpy()[0, ...]
        try:
            logging.info('Y0_est: %.4e' % y_est)
        except:
            pass
        if bsde.is_y_theoretical:
            y_anal = bsde.y_analytical(0, x0)

            y_error_abs = np.abs((y_anal - y_est))[0, ...]
            y_error_rel = np.abs((y_anal - y_est) / y_anal)[0, ...]
            try:
                logging.info('Y0_true: %.4e' % y_anal)
                logging.info('Y0_est: %.4e' % y_est)
            except:
                pass
            logging.info('relative error of Y0: %.3e' % np.abs((y_anal - y_est) / y_anal))

            np.savetxt(path_t0_sol + '/Y_0_anal.csv', y_anal)
        np.savetxt(path_t0_sol + '/Y_0_est.csv', y_est)
        logging.info("-" * 50)


        if not bsde_solver.is_nn_at_t0:
            z_est = bsde_solver.Z[0].numpy()[0, ...]
        else:
            z_est = bsde_solver.Z[0](x0, False).numpy()[0, ...]
        try:
            z_est_str = '['
            for i in range(len(z_est)):
                z_est_str += "%.3e" % z_est[i] + ', '
            z_est_str += ']'

            logging.info('Z0_est:  %s' % z_est_str)
        except:
            pass
        if bsde.is_y_theoretical:
            z_anal = bsde.z_analytical(0, x0)[0, ...]

            # print("z_est.shape", z_est.shape)
            z_anal_str = '['
            z_est_str = '['
            for i in range(len(z_anal)):
                z_anal_str += "%.3e" % z_anal[i] + ', '
                z_est_str += "%.3e" % z_est[i] + ', '
            z_anal_str += ']'
            z_est_str += ']'
            try:
                logging.info('Z0_true: %s' % z_anal_str)
                logging.info('Z0_est:  %s' % z_est_str)
            except:
                pass
            z_l2_error_abs = np.linalg.norm(z_anal - z_est)
            z_l2_error_rel = np.linalg.norm(z_anal - z_est) / np.linalg.norm(z_anal)
            logging.info("absolute L2-error of Z0: %.3e" % z_l2_error_abs)
            if np.linalg.norm(z_anal) != 0:
                logging.info('relative L2-error of Z0: %.3e' % z_l2_error_rel)

            np.savetxt(path_t0_sol + '/Z_0_anal.csv', z_anal)

        np.savetxt(path_t0_sol + '/Z_0_est.csv', z_est)
        logging.info("-" * 50)

        if not bsde_solver.is_nn_at_t0:
            gamma_est = bsde_solver.G[0].numpy()[0, ...]
        else:
            if bsde_solver.is_gamma_autodiff:
                gamma_est = bsde_solver.Z[0].jacobian_call(x0, False).numpy()[0, ...]
            else:
                gamma_est = bsde_solver.G[0](x0, False).numpy()[0, ...]
        if bsde.is_y_theoretical:
            gamma_anal = bsde.gamma_analytical(0, x0)[0, ...]

            gamma_l2_error_abs = np.linalg.norm(gamma_anal - gamma_est, ord=2, axis=(0, 1))
            gamma_l2_error_rel = np.linalg.norm(gamma_anal - gamma_est, ord=2, axis=(0, 1)) \
                                 / np.linalg.norm(gamma_anal, ord=2, axis=(0, 1))
            logging.info("absolute L2-error of Gamma0: %.3e" % gamma_l2_error_abs)
            if np.linalg.norm(gamma_anal) != 0:
                logging.info('relative L2-error of Gamma0: %.3e' % gamma_l2_error_rel)

            np.savetxt(path_t0_sol + '/Gamma_0_anal.csv', gamma_anal)

        np.save(path_t0_sol + '/Gamma_0_est.npy', gamma_est)

        logging.info("=" * 50)

        if bsde.is_y_theoretical:
            error_dict = {"Y_abs": y_error_abs, "Y_rel": y_error_rel, "Z_abs": z_l2_error_abs, "Z_rel": z_l2_error_rel,
                          "Gamma_abs": gamma_l2_error_abs, "Gamma_rel": gamma_l2_error_rel}
            with open(FLAGS.log_dir + '/Errors.csv', 'w') as csv_file:
                writer = csv.writer(csv_file)
                for key, value in error_dict.items():
                    writer.writerow([key, value])


if __name__ == '__main__':
    app.run(main)
