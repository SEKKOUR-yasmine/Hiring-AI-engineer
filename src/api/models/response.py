from fastapi.responses import JSONResponse


class PredictionResponse(JSONResponse):
    def __init__(self, prediction=None, status_code=200, headers=None, **kwargs):

        content = {
            "prediction": prediction.item(),
        }

        super().__init__(
            content=content, status_code=status_code, headers=headers, **kwargs
        )
