# SPDX-FileCopyrightText: 2020 SAP SE
#
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod

class Sanitizer(object):

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def clip_grads(self, t, name=None):
        pass

    @abstractmethod
    def add_noise(self, t, sigma, name=None):
        pass