import logging
from requests import delete, get, post

from env.core import BaseEnv

logger = logging.getLogger(__name__)


class Env(BaseEnv):

    def __init__(self, server='10.60.1.79:17101'):
        self.server = server
        self._corpus = dict()
        self._title2id = dict()

    def reset(self):
        delete(f'http://{self.server}/states')

    def get(self, p_id):
        if p_id not in self._corpus:
            self._corpus[p_id] = get(f'http://{self.server}/passages/{p_id}').json()
        return self._corpus[p_id]

    def title2id(self, norm_title):
        if norm_title not in self._title2id:
            self._title2id[norm_title] = post(f'http://{self.server}/title2id', json={"norm_title": norm_title}).json()
        return self._title2id[norm_title]

    def step(self, command, session_id=None, exclusion=None):
        args = {"command": command, "session_id": session_id, "exclusion": exclusion}
        return post(f'http://{self.server}/executions', json=args).json()
