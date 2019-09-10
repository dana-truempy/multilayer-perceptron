from Feature import Feature


class ContinuousFeature(Feature):
    """
    This class calls the parent Feature class and returns the type and distance metric for a continuous feature
    """

    def __init__(self, value):
        """
        Instantiates a new Feature of "continuous" type
        """
        super().__init__('continuous', value)

    def getDistMetric(self):
        """
        @return (feature1-feature2)^2 
        At the end, features are added together and square root is taken for Euclidean distance
        """
        return lambda feat1, feat2: (float(feat1._value) - float(feat2._value)) ** 2
