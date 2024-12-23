from enum import StrEnum
import json

class VideoClass(StrEnum):
    VARROA_FREE = "varroa_free"
    VARROA_INFESTED = "varroa_infested"

class Camera(StrEnum):
    TOP = "top"
    BOTTOM = "bottom"

class UniCalParser:
    def __init__(self, file_path: str) -> None:
        with open(file_path, "r") as f:
            self.file_lines = f.readlines()
            f.close()

    def get_frame_info(
        self, camera: Camera, video_class: VideoClass, video_id: int, frame_id: int
    ) -> tuple[bool,bool]: # isFind, VarroaVisible
        video_str = str(video_class)
        video_str += "/"
        video_str += str(video_id)

        for line in self.file_lines:
            json_line = json.loads(line)
            video_value = json_line["video"].split(" ")[0]
            if (
                video_str == video_value
                and json_line["id"] == f"frame{frame_id}"
                and json_line["camera"] == str(camera)
            ):
                # Frame detected
                if json_line["varroa_visible"] == "yes":
                    return True,True
                else:
                    return True,False
        return False, False
