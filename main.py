from fastapi import FastAPI
from pydantic import BaseModel
import os
from fastapi import FastAPI, Depends, HTTPException, Query, Request, Form, status
from typing import Annotated, Union
from datetime import datetime, timedelta, timezone

import uvicorn
from datetime import datetime, timedelta
from fastapi import Header, UploadFile, Form, File
from typing import List, Optional, Dict, Any, Annotated
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uuid
app = FastAPI()

class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float = None

@app.post("/items/")
def create_item(item: Item):
    # Process the item data (e.g., save to a database)
    return {"message": "Item created successfully", "item": item}


@app.post("/upload/sitemap")
def upload_sitemaps(
    sitemap_urls: str = Form(...)
    #request: Request=None
):
    
    
    if not sitemap_urls:
        return None

    return sitemap_urls