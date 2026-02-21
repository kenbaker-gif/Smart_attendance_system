# Quick Start - Smart Attendance API

## 1️⃣ Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file (see .env.example template)
cp .env.example .env
# Edit .env with your Supabase credentials

# Start server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Visit: http://localhost:8000/docs (Swagger UI with interactive testing)

## 2️⃣ Test Endpoints

```bash
# Run test script
python test_api.py

# Or test manually
curl http://localhost:8000/health
curl http://localhost:8000/  # See all endpoints
```

## 3️⃣ Deploy to Railway in 2 minutes

```bash
npm i -g @railway/cli
railway login
railway init
railway up
```

Get your URL: `railway status` → Use this in Flutter app

## 4️⃣ Flutter App Integration

1. Copy `/lib/services/api_service.dart` to your Flutter project
2. Update API URL in `main.dart`:
   ```dart
   const apiBaseUrl = 'https://your-railway-url.railway.app';
   ```
3. Done! See comments in `api_service.dart` for examples

## 📝 Configuration

Create `.env` file with:
```
PORT=8000
THRESHOLD=0.50
USE_SUPABASE=true
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-key
DB_USER=postgres
DB_PASSWORD=your-password
DB_HOST=your-host.supabase.com
ADMIN_SECRET=your-secret
```

## 🔍 API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | API info & endpoints |
| GET | `/health` | Health check |
| POST | `/verify` | Verify face (multipart form) |
| GET | `/attendance-records` | Get logs (optional: ?student_id=&limit=50) |
| GET | `/students` | List all students |
| POST | `/admin/sync-encodings` | Regenerate encodings (requires Authorization header) |

## 🐛 Troubleshooting

**"Not Found" error on http://localhost:8000**
- Use http://localhost:8000/docs for interactive testing
- Or try specific endpoints like `/health`

**"Recognition service not initialized"**
- Wait 30 seconds for model to load on first run
- Check logs: `tail -f logs/attendance.log`

**Database connection error**
- Verify `.env` credentials are correct
- Test connection: `python -c "from app.database import engine; engine.connect()"`

**Image upload fails**
- Ensure image is JPEG/PNG and < 5MB
- Check file permissions

## 📚 Learn More

- **API Details**: See docstrings in `app/main.py`
- **Recognition Logic**: See comments in `app/services/recognition.py`
- **Flutter Setup**: See comments in `lib/services/api_service.dart`
- **Database**: See comments in `app/database.py`
- **Deployment**: See comments in `app/main.py` module docstring

## 🚀 Deployment Checklist

- [ ] `.env` file configured with real credentials
- [ ] Test locally: `python test_api.py`
- [ ] Optional: Run admin sync to generate encodings
- [ ] Deploy to Railway: `railway up`
- [ ] Update Flutter app with Railway URL
- [ ] Test in Flutter app before releasing to store
