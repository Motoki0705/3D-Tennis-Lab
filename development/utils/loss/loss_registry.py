# filename: development/utils/loss/loss_registry.py


class LossRegistry:
    def __init__(self):
        self._loss_dict = {}

    def register(self, name, loss_class):
        self._loss_dict[name] = loss_class

    def get(self, name, **kwargs):
        if name not in self._loss_dict:
            errer_msg = f"Loss '{name}' is not registered. Available losses: {list(self._loss_dict.keys())}"
            raise ValueError(errer_msg)
        return self._loss_dict[name](**kwargs)


# グローバルなレジストリインスタンスを作成
loss_registry = LossRegistry()


# デコレータで登録を簡略化
def register_loss(name):
    def decorator(loss_class):
        loss_registry.register(name, loss_class)
        return loss_class

    return decorator
