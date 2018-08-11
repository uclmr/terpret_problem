import argparse
import sys
import time
import tensorflow as tf
from options import parse_options
from util import SharedLogDirichletInitializer
from terpret_problem import TerpretProblem

if __name__ == "__main__":
    opts = parse_options()

    with tf.Session() as sess:
        tf.set_random_seed(opts.seed)

        tp = TerpretProblem(opts)
        tf.global_variables_initializer().run(session=sess)

        for k, v in opts.__dict__.items():
            print(k, ":", v)
        sys.stdout.flush()
        time.sleep(3)

        for epoch in range(opts.n_epochs):
            _, loss = sess.run([tp.update, tp.loss])
            mus, mus_stoch = sess.run([tp.mus, tp.mus_stoch])
            sys.stdout.write("[%d] %.8f D: " % (epoch, loss))
            for i in range(opts.v):
                sys.stdout.write("%03d " % round(100 * mus[i, 0]))
            sys.stdout.write("\n")
            sys.stdout.write("[%d] %.8f S: " % (epoch, loss))
            for i in range(opts.v):
                sys.stdout.write("%03d " % round(100 * mus_stoch[i, 0]))
            sys.stdout.write("\n\n")
