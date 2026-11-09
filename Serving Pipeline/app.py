from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from predict import jambaPredictor


app = FastAPI(
    title="Data Science Assistant API",
    description="jamba-Instruct Data Science Assistant",
    version="1.0"
)


# Load model only once when API starts
predictor = jambaPredictor()


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    question: str
    answer: str


@app.get("/")
def home():

    return {
        "message": "Data Science Assistant API is running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "ai21labs/Jamba-tiny-dev"
    }


@app.post(
    "/predict",
    response_model=AnswerResponse
)
def predict(request: QuestionRequest):

    try:

        question = request.question.strip()

        if not question:
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )

        answer = predictor.predict(question)

        return {
            "question": question,
            "answer": answer
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )