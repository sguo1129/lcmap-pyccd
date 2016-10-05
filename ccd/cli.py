#!/usr/bin/env python3
""" Command line interface to Python Continuous Change Detection. """

from ccd import app
import click

logger = app.logging.getLogger(__name__)


@click.command()
@click.option('--count', default=1, help='Number of greetings.')
@click.option('--name', prompt='Your name',
              help='The person to greet.')
def detect(count, name):
    logger.info("CLI running {0} times for {1}".format(count, name))

if __name__ == '__main__':
    detect()
