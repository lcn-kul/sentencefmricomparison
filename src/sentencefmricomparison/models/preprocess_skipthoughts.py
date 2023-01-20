"""Convert the TF1 checkpoints from SkipThoughts to TF2.

Based on https://www.tensorflow.org/guide/migrate/migrating_checkpoints#checkpoint-conversion
"""

# Imports
import click
import tensorflow as tf


@click.command()
@click.option("--checkpoint-path", type=str)
@click.option("--output-prefix", type=str)
def convert_tf1_to_tf2(
    checkpoint_path: str,
    output_prefix: str,
):
    """Converts a TF1 checkpoint to TF2.

    To load the converted checkpoint, you must build a dictionary that maps
    variable names to variable objects.
    ```
    ckpt = tf.train.Checkpoint(vars={name: variable})
    ckpt.restore(converted_ckpt_path)
    ```

    Args:
      checkpoint_path: Path to the TF1 checkpoint.
      output_prefix: Path prefix to the converted checkpoint.

    Returns:
      Path to the converted checkpoint.
    """
    variables = {}
    reader = tf.train.load_checkpoint(checkpoint_path)
    dtypes = reader.get_variable_to_dtype_map()
    for key in dtypes.keys():
        variables[key] = tf.Variable(reader.get_tensor(key))
    return tf.train.Checkpoint(vars=variables).save(output_prefix)


@click.group()
def cli() -> None:
    """
    Convert the TF1 checkpoints from SkipThoughts to TF2.
    """


if __name__ == "__main__":
    cli.add_command(convert_tf1_to_tf2)
    cli()
