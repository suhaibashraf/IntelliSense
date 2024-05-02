import colorsys
from tuya_connector import TuyaOpenAPI
import tinytuya


class LightControlV2:
    def __init__(self):
        self.endpoint = "https://openapi.tuyaus.com"

        # Fill these in with your details
        self.api_key = "rrr4cgrhvckxrpc58t3w"
        self.api_secret = "3858689556d64b9d8f6ef653cd4ba2b3"
        self.device_id = "ebf94510ad14e01e99nkdp"
        self.openapi = TuyaOpenAPI(self.endpoint, self.api_key, self.api_secret)
        self.openapi.connect()
        self.commands = None

    def set_light(self, color, debug=False):
        color = self.rgb_to_hsv(color, 500)
        self.commands = {'commands': [{'code': 'colour_data_v2', 'value': {"h": color[0], "s": color[1], "v": color[2]}}]}
        if debug:
           response = self.openapi.post(f'/v1.0/iot-03/devices/{self.device_id}/commands', self.commands)
           print(response)
        else:
            self.openapi.post(f'/v1.0/iot-03/devices/{self.device_id}/commands', self.commands)

    @staticmethod
    def rgb_to_hsv(color, brightness):
        red_percentage = color[0] / float(255)
        green_percentage = color[1] / float(255)
        blue_percentage = color[2] / float(255)

        color_hsv_percentage = colorsys.rgb_to_hsv(red_percentage, green_percentage, blue_percentage)

        color_h = round(360 * color_hsv_percentage[0])
        color_s = round(1000 * color_hsv_percentage[1])
        color_v = round(brightness * color_hsv_percentage[2])
        color_hsv = [color_h, color_s, color_v]

        return color_hsv


class LightControl:
    def __init__(self):
        self.d = tinytuya.BulbDevice(
            dev_id='ebf94510ad14e01e99nkdp',
            address='192.168.8.4',
            local_key='eRkn>wxE5KzgY.3%',
            version=3.4)
        self.data = None

    def set_light(self, color):
        self.d.set_colour(color[0], color[1], color[2])

    def check_status(self):
        self.data = self.d.status()
        print(self.data)
