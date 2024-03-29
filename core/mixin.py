import ever as er
import numpy as np
import torch
import torch.nn as nn

from core import field, loss
from core.head import get_detector

MAX_TIMES = 50

class ChangeMixin(nn.Module):
    def __init__(self, feature_extractor, classifier, detector_config, loss_config):
        super(ChangeMixin, self).__init__()
        self.features = feature_extractor
        self.classifier = classifier
        self.detector_config = detector_config
        self.change_detector = get_detector(**detector_config)
        self.loss_config = er.config.AttrDict.from_dict(loss_config)

    def extract_feature(self, x):
        return self.features(x)

    def classify(self, x):
        return self.classifier(x)

    def forward(self, x, y=None):
        if self.training:
            # original_img extracted from x (x = imgs)
            x1 = x[:, :3, :, :]
            # transformed_img extracted from x (x = imgs)
            x2 = x[:, 3:, :, :]

            y1_feature = self.extract_feature(x1)
            y2_feature = self.extract_feature(x2)

            y1_pred = self.classify(y1_feature)

            # Here should be my y2_pred here
            y2_pred = self.classify(y2_feature)

            # extract positive feature
            if self.detector_config.get("t1t2", True):
                change_y1y2_logit = self.change_detector(
                    torch.cat([y1_feature, y2_feature], dim=1)
                )
            else:
                change_y1y2_logit = None
            if self.detector_config.get("t2t1", True):
                change_y2y1_logit = self.change_detector(
                    torch.cat([y2_feature, y1_feature], dim=1)
                )
            else:
                change_y2y1_logit = None


            # TODO Change the field vlaue shere accordingly
            y1_true = y['change'][:, :3, :, :]
            y2_true = y['change'][:, 3:, :, :]

            loss_dict = dict()
            loss_dict.update(loss.misc_info(y1_pred.device))

            if self.detector_config.get("symmetry_loss", False):
                loss_dict.update(
                    loss.semantic_and_symmetry_loss(
                        y1_true,
                        y2_true,
                        y1_pred,
                        y2_pred,
                        change_y1y2_logit,
                        change_y2y1_logit,
                        self.loss_config,
                    )
                )
            else:
                raise ValueError()

            return loss_dict

        x1 = x[:, :3, :, :]
        x2 = x[:, 3:, :, :]

        y1_feature = self.extract_feature(x1)
        y2_feature = self.extract_feature(x2)

        y1_pred = self.classify(y1_feature)

        change_y1y2_logit = self.change_detector(
            torch.cat([y1_feature, y2_feature], dim=1)
        )

        y2_pred = self.classify(y2_feature)

        logit1 = y1_pred
        logit2 = y2_pred
        change_logit = change_y1y2_logit
        return torch.cat([logit1, logit2, change_logit], dim=1)
