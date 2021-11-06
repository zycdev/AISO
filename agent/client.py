import logging
from requests import delete, get, post

logger = logging.getLogger(__name__)


class Agent(object):

    def __init__(self, server='10.60.1.79:17101'):
        self.server = server

    def reset(self):
        delete(f'http://{self.server}/games')

    def memory(self, game_id, dtype='dict'):
        return get(f'http://{self.server}/memory/{game_id}', params={"dtype": dtype}).json()

    def add_evidence(self, game_id, p_id):
        post(f'http://{self.server}/memory/{game_id}/{p_id}')

    def del_evidence(self, game_id, p_id):
        delete(f'http://{self.server}/memory/{game_id}/{p_id}')

    def act(self, game_ids, questions, observations=None, review=False):
        args = {"game_ids": game_ids, "questions": questions, "observations": observations, "review": review}
        return post(f'http://{self.server}/actions', json=args).json()

    def proposals(self, game_id, step):
        return get(f'http://{self.server}/proposals/{game_id}/{step}').json()
