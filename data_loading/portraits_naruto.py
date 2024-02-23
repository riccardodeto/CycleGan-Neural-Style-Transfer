import tensorflow as tf
import os
import tensorflow_datasets as tfds


# Saving the dataset to tensorflow
class PortraitsNarutoConfig(tfds.core.BuilderConfig):
    def __init__(self, *, split=None, **kwargs):
        super(PortraitsNarutoConfig, self).__init__(**kwargs)
        self.split = split


class PortraitsNaruto(tfds.core.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        PortraitsNarutoConfig(
            name="portraits_naruto",
            version=tfds.core.Version("1.0.0"),
            description="Dataset containing images from the Anime Naruto and portraits.",
        ),
    ]

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(),
                "label": tfds.features.ClassLabel(names=["A", "B"]),
            }),
            supervised_keys=("image", "label"),
            homepage="",
            citation="",
        )

    def _split_generators(self, dl_manager):
        dataset_path = '/content/dataset'
        train_a_path = os.path.join(dataset_path, 'trainA')
        train_b_path = os.path.join(dataset_path, 'trainB')
        test_a_path = os.path.join(dataset_path, 'testA')
        test_b_path = os.path.join(dataset_path, 'testB')
        val_a_path = os.path.join(dataset_path, 'valA')
        val_b_path = os.path.join(dataset_path, 'valB')

        return [
            tfds.core.SplitGenerator(
                name="trainA",
                gen_kwargs={
                    "path": train_a_path,
                    "label": "A",
                },
            ),
            tfds.core.SplitGenerator(
                name="trainB",
                gen_kwargs={
                    "path": train_b_path,
                    "label": "B",
                },
            ),
            tfds.core.SplitGenerator(
                name="testA",
                gen_kwargs={
                    "path": test_a_path,
                    "label": "A",
                },
            ),
            tfds.core.SplitGenerator(
                name="testB",
                gen_kwargs={
                    "path": test_b_path,
                    "label": "B",
                },
            ),
            tfds.core.SplitGenerator(
                name="valA",
                gen_kwargs={
                    "path": val_a_path,
                    "label": "A",
                },
            ),
            tfds.core.SplitGenerator(
                name="valB",
                gen_kwargs={
                    "path": val_b_path,
                    "label": "B",
                },
            ),
        ]

    def _generate_examples(self, path, label):
        images = tf.io.gfile.listdir(path)

        for image in images:
            record = {
                "image": os.path.join(path, image),
                "label": label,
            }
            yield image, record