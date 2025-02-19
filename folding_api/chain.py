import bittensor as bt
from loguru import logger


class SubtensorService:
    def __init__(self, config=None):
        self.config = config
        # Initialize subtensor with config
        self.subtensor = bt.subtensor(config=self.config) if config else bt.subtensor()
        self.metagraph = self.subtensor.metagraph(self.config.netuid if config else 25)
        self.wallet = bt.wallet(config=self.config) if config else bt.wallet()
        self.dendrite = bt.dendrite(wallet=self.wallet)

    def resync_metagraph(self):
        self.metagraph.sync(subtensor=self.subtensor)
        logger.info("metagraph_reloaded")
