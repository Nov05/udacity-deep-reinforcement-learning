import logging

from communicator_objects import UnityOutput, UnityInput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("unityagents")


class Communicator(object):
    def __init__(self, worker_id=0,
                 base_port=5005):
        """
        Python side of the communication. Must be used in pair with the right Unity Communicator equivalent.

        :int base_port: Baseline port number to connect to Unity environment over. worker_id increments over this.
        :int worker_id: Number to add to communication port (5005) [0]. Used for asynchronous agent scenarios.
        """

    def initialize(self, inputs: UnityInput) -> UnityOutput: # type: ignore
        """
        Used to exchange initialization parameters between Python and the Environment
        :param inputs: The initialization input that will be sent to the environment.
        :return: UnityOutput: The initialization output sent by Unity
        """

    def exchange(self, inputs: UnityInput) -> UnityOutput: # type: ignore
        """
        Used to send an input and receive an output from the Environment
        :param inputs: The UnityInput that needs to be sent the Environment
        :return: The UnityOutputs generated by the Environment
        """

    def close(self):
        """
        Sends a shutdown signal to the unity environment, and closes the connection.
        """

