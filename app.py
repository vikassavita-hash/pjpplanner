import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from pjp_balancer import generate_dynamic_pjp

app = FastAPI(title="PJP Planner API Server")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the main PJP Planner frontend index.html"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read index.html: {str(e)}")

@app.post("/api/generate-pjp")
async def api_generate_pjp(
    file: UploadFile = File(...),
    num_merch: int = Form(default=9),
    format_type: str = Form(default="excel")
):
    """
    Accepts a store master CSV or Excel file, validates and clusters outlets
    geographically, spreads visits evenly across 24 days, and returns 
    a styled multi-tab Excel spreadsheet for download or JSON structure for local portal loading.
    """
    # Validate headcount input
    if num_merch < 9:
        raise HTTPException(status_code=400, detail="Total merchandisers headcount must be 9 or more.")

    filename = file.filename.lower()
    contents = await file.read()
    
    # Parse the uploaded file into a Pandas DataFrame
    try:
        if filename.endswith(".csv"):
            df_input = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith((".xlsx", ".xls")):
            df_input = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file format. Please upload a valid .csv, .xlsx, or .xls file."
            )
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Failed to parse uploaded file: {str(e)}"
        )

    # Execute the PJP VRP balancing pipeline
    try:
        pjp_result = generate_dynamic_pjp(df_input, num_merch, format_type=format_type)
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Optimization algorithm failed: {str(e)}"
        )
        
    if format_type == "json":
        return pjp_result
        
    # Stream the styled spreadsheet back to the client
    return StreamingResponse(
        io.BytesIO(pjp_result),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": "attachment; filename=PJP_Balanced_Matrix.xlsx"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
