from .storage import Storage


class DummyStorage(Storage):
    """
    Dummy Loud ML storage for testing
    """

    def model_exists(self, name):
        return False

    def get_model_data(self, name, ckpt_name=None):
        return {}

    def get_template_data(self, name):
        return {}

    def list_models(self):
        return []

    def list_checkpoints(self, name):
        return []

    def list_templates(self):
        return []

    def create_model(self, model, config):
        pass

    def delete_model(self, name):
        pass

    def save_model(self, model, save_state=True):
        pass

    def save_state(self, model, ckpt_name=None):
        pass

    def set_current_ckpt(self, model_name, ckpt_name):
        pass

    def get_current_ckpt(self, model_name):
        return None

    def load_model(self, name, ckpt_name=None):
        return None

    def load_template(self, _name, *args, **kwargs):
        return None

    def get_model_hook(self, model_name, hook_name):
        return None

    def list_model_hooks(self, model_name):
        return []

    def set_model_hook(self, model_name, hook_name, hook_type, config=None):
        pass

    def delete_model_hook(self, model_name, hook_name):
        pass
