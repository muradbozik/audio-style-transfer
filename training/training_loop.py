from __future__ import print_function, division
import tensorflow as tf

from training.networks import extract_image, assemble_image
from training.loss import Losses


# Training Functions

# Train Generator, Siamese and Critic
@tf.function
def train_all(a, b, args):
    # splitting spectrogram in 3 parts
    aa, aa2, aa3 = extract_image(a)
    bb, bb2, bb3 = extract_image(b)
    L = Losses(args.delta)

    with tf.GradientTape() as tape_gen, tf.GradientTape() as tape_disc:
        # translating A to B
        fab = args.gen(aa, training=True)
        fab2 = args.gen(aa2, training=True)
        fab3 = args.gen(aa3, training=True)
        # identity mapping B to B     COMMENT THESE 3 LINES IF THE IDENTITY LOSS TERM IS NOT NEEDED
        fid = args.gen(bb, training=True)
        fid2 = args.gen(bb2, training=True)
        fid3 = args.gen(bb3, training=True)
        # concatenate/assemble converted spectrograms
        fabtot = assemble_image([fab, fab2, fab3])

        # feed concatenated spectrograms to critic
        cab = args.critic(fabtot, training=True)
        cb = args.critic(b, training=True)
        # feed 2 pairs (A,G(A)) extracted spectrograms to Siamese
        sab = args.siam(fab, training=True)
        sab2 = args.siam(fab3, training=True)
        sa = args.siam(aa, training=True)
        sa2 = args.siam(aa3, training=True)

        # identity mapping loss
        loss_id = (L.mae(bb, fid) + L.mae(bb2, fid2) + L.mae(bb3, fid3)) / \
                  3.  # loss_id = 0. IF THE IDENTITY LOSS TERM IS NOT NEEDED
        # travel loss
        loss_m = L.loss_travel(sa, sab, sa2, sab2) + L.loss_siamese(sa, sa2)
        # generator and critic losses
        loss_g = L.g_loss_f(cab)
        loss_dr = L.d_loss_r(cb)
        loss_df = L.d_loss_f(cab)
        loss_d = (loss_dr + loss_df) / 2.
        # generator+siamese total loss
        # CHANGE LOSS WEIGHTS HERE  (COMMENT OUT +w*loss_id IF THE IDENTITY LOSS TERM IS NOT NEEDED)
        lossgtot = loss_g + 10. * loss_m + 0.5 * loss_id

    # computing and applying gradients
    grad_gen = tape_gen.gradient(
        lossgtot, args.gen.trainable_variables + args.siam.trainable_variables)
    args.opt_gen.apply_gradients(
        zip(grad_gen, args.gen.trainable_variables + args.siam.trainable_variables))

    grad_disc = tape_disc.gradient(loss_d, args.critic.trainable_variables)
    args.opt_disc.apply_gradients(zip(grad_disc, args.critic.trainable_variables))

    return loss_dr, loss_df, loss_g, loss_id


# Train Critic only
@tf.function
def train_d(a, b, args):
    aa, aa2, aa3 = extract_image(a)
    L = Losses(args.delta)
    with tf.GradientTape() as tape_disc:
        fab = args.gen(aa, training=True)
        fab2 = args.gen(aa2, training=True)
        fab3 = args.gen(aa3, training=True)
        fabtot = assemble_image([fab, fab2, fab3])

        cab = args.critic(fabtot, training=True)
        cb = args.critic(b, training=True)

        loss_dr = L.d_loss_r(cb)
        loss_df = L.d_loss_f(cab)

        loss_d = (loss_dr + loss_df) / 2.

    grad_disc = tape_disc.gradient(loss_d, args.critic.trainable_variables)
    args.opt_disc.apply_gradients(zip(grad_disc, args.critic.trainable_variables))

    return loss_dr, loss_df
