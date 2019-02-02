"""
Loud ML file storage
"""

import loudml.vendor

import copy
import glob
import json
import logging
import os
import shutil
import tempfile

from voluptuous import (
    All,
    Length,
    Match,
)

from . import (
    errors,
    schemas,
)

from .storage import (
    Storage,
)

from dictdiffer import diff

OBJECT_KEY_SCHEMA = schemas.All(
   str,
   Length(min=1),
   Match("^[a-zA-Z0-9-_@.]+$"),
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

    def get_ckpt_name(self, i):
        return "{:02d}".format(i)

    def get_next_ckpt_name(self, model_path):
        path = next(iter(sorted(glob.glob(os.path.join(model_path, '*.ckpt')),
            key=lambda f: os.stat(f).st_mtime, reverse=True)), None)

        if path is None:
            return self.get_ckpt_name(0)
        else:
            ckpt = os.path.splitext(os.path.basename(path))[0]
            return self.get_ckpt_name(int(ckpt)+1)

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
        tmp_fd, tmp_path = tempfile.mkstemp(prefix=path + ".");
        with open(tmp_path, 'w') as fd:
            json.dump(data, fd)
            os.fsync(fd)
        os.chmod(tmp_path, 0o660)
        os.rename(tmp_path, path)
        os.close(tmp_fd)

    def _load_json(self, path):
        with open(path) as fd:
            return json.load(fd)

    def _write_model_settings(self, model_path, settings):
        settings = copy.deepcopy(settings)
        settings.pop('name', None)
        self._write_json(os.path.join(model_path, "settings.json"), settings)

    def _write_model_state(self, model_path, state=None, ckpt_name=None):
        if ckpt_name is None:
            try:
                state_path = os.readlink(os.path.join(model_path, "state.json"))
            except OSError:
                # convert state.json file
                ckpt_name = self.get_ckpt_name(0)
                state_path = os.path.join(model_path, "{}.ckpt".format(ckpt_name))
                os.rename(os.path.join(model_path, "state.json"), state_path)
        else:
            state_path = os.path.join(model_path, "{}.ckpt".format(ckpt_name))

        if state is None:
            try:
                os.unlink(state_path)
            except FileNotFoundError:
                pass
        else:
            self._write_json(state_path, state)

    def _write_model(self, path, settings, state=None, save_state=True, save_ckpt=True):
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as exn:
            raise errors.LoudMLException(str(exn))

        self._write_model_settings(path, settings)
        if save_state:
            ckpt_name = None
            if save_ckpt:
                ckpt_name = self.get_next_ckpt_name(path)

            self._write_model_state(path, state, ckpt_name)
            if save_ckpt:
                self._set_current_ckpt(path, ckpt_name)

    def create_model(self, model, config=None):
        model_path = self.model_path(model.name)

        if os.path.exists(model_path):
            raise errors.ModelExists()

        self._write_model(model_path, model.settings, model.state, save_state=False)

    def save_model(self, model, save_state=True, save_ckpt=True):
        model_path = self.model_path(model.name)
        try:
            old_settings = self._get_model_settings(model_path, model.name)
        except errors.LoudMLException as exn:
            old_settings = {}

        old_settings['name'] = model.name

        self._write_model(
            model_path,
            model.settings,
            model.state,
            save_state,
            save_ckpt,
        )
        return diff(old_settings, model.settings, expand=True)

    def save_state(self, model, ckpt_name=None):
        self._write_model_state(self.model_path(model.name), model.state, ckpt_name)

    def _set_current_ckpt(self, model_path, ckpt_name):
        state_path = os.path.join(model_path, "state.json")
        ckpt_path = os.path.join(model_path, "{}.ckpt".format(ckpt_name))
        try:
            os.unlink(state_path)
        except FileNotFoundError:
            pass

        os.symlink(ckpt_path, state_path)
        # touch the file to update mtime
        with open(ckpt_path, 'a'):
            os.utime(ckpt_path, None)

    def set_current_ckpt(self, model_name, ckpt_name):
        model_path = self.model_path(model_name)
        self._set_current_ckpt(model_path, ckpt_name)

    def delete_model(self, name):
        try:
            shutil.rmtree(self.model_path(name))
        except FileNotFoundError:
            raise errors.ModelNotFound(name=name)

    def model_exists(self, name):
        return os.path.exists(self.model_path(name))

    def _get_model_settings(self, model_path, model_name):
        settings_path = os.path.join(model_path, "settings.json")
        try:
            return self._load_json(settings_path)
        except ValueError as exn:
            raise errors.Invalid(
                "invalid model setting file: {}: {}".format(
                    settings_path,
                    str(exn),
                )
            )
        except FileNotFoundError:
            raise errors.ModelNotFound(name=model_name)
        except OSError as exn:
            raise errors.LoudMLException(str(exn))

    def _get_model_state(self, model_path, ckpt_name=None):
        if ckpt_name is None:
            state_path = os.path.join(model_path, "state.json")
        else:
            state_path = os.path.join(model_path, "{}.ckpt".format(ckpt_name))

        try:
            return self._load_json(state_path)
        except ValueError as exn:
            raise errors.Invalid(
                "invalid model state file: {}: {}".format(
                    state_path,
                    str(exn),
                )
            )
        except FileNotFoundError:
            # Model is not trained yet
            return None

    def get_model_data(self, name, ckpt_name=None):
        model_path = self.model_path(name)
        settings = self._get_model_settings(model_path, name)
        settings['name'] = name

        data = {
            'settings': settings,
        }

        try:
            state = self._get_model_state(model_path, ckpt_name)
            if state is not None:
                data['state'] = state
        except errors.Invalid as exn:
            logging.error(str(exn))

            # XXX: Keep broken file for troubleshooting
            state_path = os.path.join(model_path, "state.json")
            os.rename(state_path, state_path + ".broken")

        return data

    def list_checkpoints(self, name):
        return sorted([
            os.path.splitext(os.path.basename(path))[0]
            for path in glob.glob(os.path.join(self.model_dir, name, '*.ckpt'))
        ])

    def list_models(self):
        return sorted([
            os.path.splitext(os.path.basename(path))[0]
            for path in glob.glob(self.model_path('*', validate=False))
        ])

    def _write_model_hook(self, model_name, settings):
        model_path = self.model_path(model_name)
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
            for path in glob.glob(self._hook_path(hooks_dir, '*',
                                                  validate=False))
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
            raise errors.ModelNotFound(name=model_name)

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

    def _build_object_path(self, model_name, key):
        """Build the path of a model object"""

        model_path = self.model_path(model_name)
        schemas.validate(OBJECT_KEY_SCHEMA, key)
        return os.path.join(model_path, "objects", key + ".json")


    def set_model_object(self, model_name, key, data):
        """Save model object"""

        path = self._build_object_path(model_name, key)

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except OSError as exn:
            raise KeyError("model object not found")

        self._write_json(path, data)

    def get_model_object(self, model_name, key):
        """Get model object"""

        path = self._build_object_path(model_name, key)

        try:
            return self._load_json(path)
        except ValueError as exn:
            raise errors.Invalid("invalid object: file: %s", str(exn))
        except FileNotFoundError:
            raise KeyError("model object not found")

    def delete_model_object(self, model_name, key):
        """Delete model object"""

        path = self._build_object_path(model_name, key)

        try:
            os.unlink(path)
        except FileNotFoundError:
            raise KeyError("model object not found")


class TempStorage(FileStorage):
    """
    Temporary file storage
    """

    def __init__(self, prefix="", suffix=""):
        self.tmp_dir = tempfile.mkdtemp(prefix=prefix, suffix=suffix)
        super().__init__(self.tmp_dir)

    def __del__(self):
        shutil.rmtree(self.tmp_dir)
