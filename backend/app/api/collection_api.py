from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict
from pydantic import BaseModel
import logging

from backend.app.services.vstore_svc import VectorStoreService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/collections", tags=["Collections"])

class CollectionResponse(BaseModel):
    name: str
    document_count: int

class CollectionCreateRequest(BaseModel):
    name: str

def get_vector_store_service():
    return VectorStoreService()

@router.post("", response_model=CollectionResponse)
async def create_collection(
    request: CollectionCreateRequest,
    vstore_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """Create a new collection."""
    logger.info(f"Received request to create collection: {request.name}")
    try:
        logger.info(f"Calling VectorStoreService.create_collection with name: {request.name}")
        vstore_svc.create_collection(request.name)
        logger.info(f"Successfully created collection: {request.name}")
        return CollectionResponse(
            name=request.name,
            document_count=0
        )
    except ValueError as e:
        logger.error(f"ValueError while creating collection: {str(e)}")
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error while creating collection: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to create collection: {str(e)}")

@router.get("", response_model=List[CollectionResponse])
async def list_collections(
    vstore_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """List all collections."""
    try:
        collections = vstore_svc.list_collections()
        return [
            CollectionResponse(
                name=name,
                document_count=count
            ) for name, count in collections
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")

@router.delete("/{name}")
async def delete_collection(
    name: str,
    vstore_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """Delete a collection."""
    try:
        success = vstore_svc.delete_collection(name)
        if not success:
            raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
        return {"message": f"Collection '{name}' deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete collection: {str(e)}") 