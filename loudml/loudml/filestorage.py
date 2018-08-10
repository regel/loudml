"""
LoudML file storage
"""

import copy
import glob
import json
import logging
import os
import shutil
import tempfile

from . import (
    errors,
    schemas,
)

from .storage import (
    Storage,
)

class FileStorage(Storage):
    """
    File storage
    """

    def __init__(self, path):
        self.path = path
        self.model_dir = os.path.join(path, 'models')

        try:
            os.makedirs(self.model_dir, exist_ok=True)
        except OSError as exn:
            raise errors.LoudMLException(str(exn))

        self._convert_models()

    def _convert_models(self):
        """
        Convert single-file model to the new format
        """
        for path in glob.glob(os.path.join(self.model_dir, '*.lmm')):
            model_name = os.path.splitext(os.path.basename(path))[0]
            logging.info("converting model `%s' to the new format", model_name)

            try:
                with open(path) as model_file:
                    data = json.load(model_file)
            except ValueError as exn:
                logging.error("invalid model file: %s", str(exn))

            self._write_model(
                self.model_path(model_name),
                data['settings'],
                data['state'],
            )
            os.unlink(path)

    def model_path(self, model_name, validate=True):
        """
        Build model path
        """
        if validate:
            schemas.validate(schemas.key, model_name, name='model_name')
        return os.path.join(self.model_dir, model_name)

    def _write_json(self, path, data):
        tmp_path = path + ".tmp"

        with open(tmp_path, 'w') as fd:
            json.dump(data, fd)

        os.rename(tmp_path, path)

    def _load_json(self, path):
        with open(path) as fd:
            return json.load(fd)

    def _write_model_settings(self, model_path, settings):
        settings = copy.deepcopy(settings)
        settings.pop('name', None)
        self._write_json(os.path.join(model_path, "settings.json"), settings)

    def _write_model_state(self, model_path, state=None):
        state_path = os.path.join(model_path, "state.json")

        if state is None:
            try:
                os.unlink(state_path)
            except FileNotFoundError as exn:
                pass
        else:
            self._write_json(state_path, state)

    def _write_model(self, path, settings, state=None):
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as exn:
            raise errors.LoudMLException(str(exn))

        self._write_model_settings(path, settings)
        self._write_model_state(path, state)

    def create_model(self, model, limits=None):
        if limits and len(_self.list_models()) >= limits['nrmodels']:
            raise errors.Forbidden(
                    "You've reached the maximum count allowed in your license")

        model_path = self.model_path(model.name)

        if os.path.exists(model_path):
            raise errors.ModelExists()

        self._write_model(model_path, model.settings, model.state)

    def save_model(self, model):
        self._write_model(self.model_path(model.name), model.settings, model.state)

    def delete_model(self, name):
        try:
            shutil.rmtree(self.model_path(name))
        except FileNotFoundError:
            raise errors.ModelNotFound()

    def model_exists(self, name):
        return os.path.exists(self.model_path(name))

    def _get_model_settings(self, model_path):
        settings_path = os.path.join(model_path, "settings.json")
        try:
            return self._load_json(settings_path)
        except ValueError as exn:
            raise errors.Invalid("invalid model setting file: %s", str(exn))
        except FileNotFoundError:
            raise errors.ModelNotFound()

    def _get_model_state(self, model_path):
        state_path = os.path.join(model_path, "state.json")
        try:
            return self._load_json(state_path)
        except ValueError as exn:
            raise errors.Invalid("invalid model state file: %s", str(exn))
        except FileNotFoundError:
            # Model is not trained yet
            return None

    def get_model_data(self, name):
        model_path = self.model_path(name)
        settings = self._get_model_settings(model_path)
        settings['name'] = name

        data = {
            'settings': settings,
        }

        state = self._get_model_state(model_path)
        if state is not None:
            data['state'] = state

        return data

    def list_models(self):
        return [
             os.path.splitext(os.path.basename(path))[0]
             for path in glob.glob(self.model_path('*', validate=False))
        ]

    def _write_model_hook(self, model_name, settings):
        settings = copy.deepcopy(settings)
        settings.pop('name', None)
        self._write_json(os.path.join(model_path, "settings.json"), settings)

    def _hook_path(self, hooks_dir, hook_name, validate=True):
        if validate:
            schemas.validate(schemas.key, hook_name)
        return os.path.join(hooks_dir, "{}.json".format(hook_name))

    def model_hooks_dir(self, model_name):
        """
        Build path to model hooks directory
        """
        return os.path.join(self.model_path(model_name), "hooks")

    def list_model_hooks(self, model_name):
        """List model hooks"""

        hooks_dir = self.model_hooks_dir(model_name)

        return [
             os.path.splitext(os.path.basename(path))[0]
             for path in glob.glob(self._hook_path(hooks_dir, '*', validate=False))
        ]

    def get_model_hook(self, model_name, hook_name):
        """Get model hook"""

        hooks_dir = self.model_hooks_dir(model_name)
        hook_path = self._hook_path(hooks_dir, hook_name)

        try:
            return self._load_json(hook_path)
        except ValueError as exn:
            raise errors.Invalid("invalid model hook file: %s", str(exn))
        except FileNotFoundError:
            raise errors.NotFound("hook not found")

    def set_model_hook(self, model_name, hook_name, hook_type, config):
        """Set model hook"""

        if not self.model_exists(model_name):
            raise errors.ModelNotFound()

        hooks_dir = self.model_hooks_dir(model_name)
        try:
            os.makedirs(hooks_dir, exist_ok=True)
        except OSError as exn:
            raise errors.LoudMLException(str(exn))

        self._write_json(self._hook_path(hooks_dir, hook_name), {
            'type': hook_type,
            'config': config,
        })

    def delete_model_hook(self, model_name, hook_name):
        """Delete model hook"""
        hooks_dir = self.model_hooks_dir(model_name)
        hook_path = self._hook_path(hooks_dir, hook_name)

        try:
            os.unlink(hook_path)
        except FileNotFoundError:
            raise errors.NotFound("hook not found")


class TempStorage(FileStorage):
    """
    Temporary file storage
    """

    def __init__(self, prefix="", suffix=""):
        self.tmp_dir = tempfile.mkdtemp(prefix=prefix, suffix=suffix)
        super().__init__(self.tmp_dir)

    def __del__(self):
        shutil.rmtree(self.tmp_dir)
