import tensorflow as tf
from architecture.losses import *

class Iterations():
    def __init__(self, generator_pn, generator_np, discriminator_p, discriminator_n, generator_pn_optimizer, generator_np_optimizer, discriminator_p_optimizer, discriminator_n_optimizer):
        self.generator_pn = generator_pn
        self.generator_np = generator_np
        self.discriminator_p = discriminator_p
        self.discriminator_n = discriminator_n
        self.generator_pn_optimizer = generator_pn_optimizer
        self.generator_np_optimizer = generator_np_optimizer
        self.discriminator_p_optimizer = discriminator_p_optimizer
        self.discriminator_n_optimizer = discriminator_n_optimizer


    @tf.function
    def train_step(self, real_p, real_n):
        with tf.GradientTape(persistent=True) as tape:
            fake_n = self.generator_pn(real_p, training=True)
            cycled_p = self.generator_np(fake_n, training=True)

            fake_p = self.generator_np(real_n, training=True)
            cycled_n = self.generator_pn(fake_p, training=True)

            same_p = self.generator_np(real_p, training=True)
            same_n = self.generator_pn(real_n, training=True)

            disc_real_p = self.discriminator_p(real_p, training=True)
            disc_real_n = self.discriminator_n(real_n, training=True)

            disc_fake_p = self.discriminator_p(fake_p, training=True)
            disc_fake_n = self.discriminator_n(fake_n, training=True)

            gen_pn_loss = generator_loss(disc_fake_n)
            gen_np_loss = generator_loss(disc_fake_p)

            cycle1 = calc_cycle_loss(real_p, cycled_p)
            cycle2 = calc_cycle_loss(real_n, cycled_n)

            total_cycle_loss = cycle1 + cycle2

            identity1 = identity_loss(real_p, same_p)
            identity2 = identity_loss(real_n, same_n)

            total_gen_pn_loss = gen_pn_loss + total_cycle_loss + identity2
            total_gen_np_loss = gen_np_loss + total_cycle_loss + identity1

            disc_p_loss = discriminator_loss(disc_real_p, disc_fake_p)
            disc_n_loss = discriminator_loss(disc_real_n, disc_fake_n)

        generator_pn_gradients = tape.gradient(total_gen_pn_loss,
                                               self.generator_pn.trainable_variables)
        generator_np_gradients = tape.gradient(total_gen_np_loss,
                                               self.generator_np.trainable_variables)

        discriminator_n_gradients = tape.gradient(disc_n_loss,
                                                  self.discriminator_n.trainable_variables)

        discriminator_p_gradients = tape.gradient(disc_p_loss,
                                                  self.discriminator_p.trainable_variables)

        self.generator_pn_optimizer.apply_gradients(zip(generator_pn_gradients,
                                                   self.generator_pn.trainable_variables))

        self.generator_np_optimizer.apply_gradients(zip(generator_np_gradients,
                                                   self.generator_np.trainable_variables))

        self.discriminator_n_optimizer.apply_gradients(zip(discriminator_n_gradients,
                                                      self.discriminator_n.trainable_variables))

        self.discriminator_p_optimizer.apply_gradients(zip(discriminator_p_gradients,
                                                      self.discriminator_p.trainable_variables))

        return [total_gen_pn_loss, total_gen_np_loss, disc_p_loss, disc_n_loss, gen_pn_loss, gen_np_loss, cycle1, cycle2, identity1, identity2]

    # @tf.function
    def val_step(self, real_p, real_n):
        with tf.GradientTape(persistent=True) as tape:
            fake_n = self.generator_pn(real_p, training=False)
            cycled_p = self.generator_np(fake_n, training=False)

            fake_p = self.generator_np(real_n, training=False)
            cycled_n = self.generator_pn(fake_p, training=False)

            same_p = self.generator_np(real_p, training=False)
            same_n = self.generator_pn(real_n, training=False)

            disc_real_p = self.discriminator_p(real_p, training=False)
            disc_real_n = self.discriminator_n(real_n, training=False)

            disc_fake_p = self.discriminator_p(fake_p, training=False)
            disc_fake_n = self.discriminator_n(fake_n, training=False)

            gen_pn_loss = generator_loss(disc_fake_n)
            gen_np_loss = generator_loss(disc_fake_p)

            cycle1 = calc_cycle_loss(real_p, cycled_p)
            cycle2 = calc_cycle_loss(real_n, cycled_n)

            total_cycle_loss = cycle1 + cycle2

            identity1 = identity_loss(real_p, same_p)
            identity2 = identity_loss(real_n, same_n)

            total_gen_pn_loss = gen_pn_loss + total_cycle_loss + identity2
            total_gen_np_loss = gen_np_loss + total_cycle_loss + identity1

            disc_p_loss = discriminator_loss(disc_real_p, disc_fake_p)
            disc_n_loss = discriminator_loss(disc_real_n, disc_fake_n)

        return [total_gen_pn_loss, total_gen_np_loss, disc_p_loss, disc_n_loss, gen_pn_loss, gen_np_loss, cycle1, cycle2, identity1, identity2]
