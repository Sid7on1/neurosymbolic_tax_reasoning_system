import logging
from typing import Dict
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from main_inference import TaxCalculator
from logging.config import dictConfig
import uvicorn
import json

# Configure logging
dictConfig({
    'version': 1,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
            'formatter': 'default'
        }
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['console']
    }
})

# Initialize logger
logger = logging.getLogger(__name__)

# Define configuration class
class Config:
    def __init__(self, tax_calculator: TaxCalculator):
        self.tax_calculator = tax_calculator

# Define request and response models
class TaxCalculationRequest(BaseModel):
    income: float
    deductions: float
    credits: float

class TaxCalculationResponse(BaseModel):
    tax_owed: float
    refund: float

# Define exception classes
class InvalidRequestError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

class InternalServerError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

# Define the main application class
class TaxCalculationService:
    def __init__(self, config: Config):
        self.config = config
        self.app = FastAPI()

    def create_app(self):
        @self.app.post("/tax_calculation", response_model=TaxCalculationResponse)
        async def tax_calculation_endpoint(request: TaxCalculationRequest):
            try:
                tax_owed, refund = self.config.tax_calculator.calculate_tax(request.income, request.deductions, request.credits)
                return TaxCalculationResponse(tax_owed=tax_owed, refund=refund)
            except Exception as e:
                logger.error(f"Error calculating tax: {str(e)}")
                raise HTTPException(status_code=500, detail="Internal Server Error")

        @self.app.get("/health_check")
        async def health_check():
            return {"status": "healthy"}

        @self.app.exception_handler(HTTPException)
        async def error_handler(request: Request, exc: HTTPException):
            logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
            return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

        @self.app.exception_handler(InvalidRequestError)
        async def invalid_request_error_handler(request: Request, exc: InvalidRequestError):
            logger.error(f"Invalid Request Error: {exc.message}")
            return JSONResponse(status_code=400, content={"error": exc.message})

        @self.app.exception_handler(InternalServerError)
        async def internal_server_error_handler(request: Request, exc: InternalServerError):
            logger.error(f"Internal Server Error: {exc.message}")
            return JSONResponse(status_code=500, content={"error": "Internal Server Error"})

        return self.app

# Create the application
def main():
    tax_calculator = TaxCalculator()
    config = Config(tax_calculator)
    service = TaxCalculationService(config)
    app = service.create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()