import torch
import wandb
import os

from ganground.utils import SingletonType

class Wandb(object, metaclass=SingletonType):

    def __init__(self, name, id, project) :
        self.api_key = "eaa5a2cd7e8a97eb4a9c19a86b93b38d7c704f69"
        self.entity = "anirbanl"
        os.environ["WANDB_API_KEY"] = self.api_key
        os.environ["WANDB_ENTITY"] = self.entity
        wandb.login(key=self.api_key)

        self.init_tracking(name, id, project)

    def init_tracking(self, name, id, project):
        self.name = name
        self.project = project
        self.id = id
        wandb.init(name=self.name, entity=self.entity, project=self.project, id=self.id)
        self.purged = False

    def log(self, map, step=None):
        assert (not self.purged)
        wandb.log(map, step)

    def set_config(self, args):
        wandb.config.update(args)

    def save(self):
        wandb.save('%s.h5'%(self.name))
        self.purged = True

    def restore(self):
        wandb.restore('%s.h5'%(self.name))
        self.purged = False

    def watch(self, model_tuple):
        assert (not self.purged)
        wandb.watch(model_tuple, log='all')
