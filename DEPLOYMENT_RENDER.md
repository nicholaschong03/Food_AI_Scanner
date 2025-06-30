# Deploy Food AI API to Render

## Why Render?
- **Free forever**: 750 hours/month (enough for 24/7 deployment)
- **Easy setup**: Automatic deployments from GitHub
- **No credit card required**: For free tier
- **Custom domains**: Free SSL certificates
- **Sleep behavior**: Goes to sleep after 15 minutes of inactivity, wakes up on request

## Prerequisites
1. GitHub account with your code pushed
2. Render account (free)
3. Google API key for Gemini Vision AI

## Step-by-Step Deployment

### 1. Prepare Your Repository
Make sure your code is pushed to GitHub with these files:
- `app.py` (main FastAPI application)
- `requirements.txt` (dependencies)
- `render.yaml` (deployment config)
- `models/` folder (your ML models)

### 2. Create Render Account
1. Go to [render.com](https://render.com)
2. Sign up with GitHub (recommended)
3. Verify your email

### 3. Deploy Your Service
1. **Connect GitHub**: Click "New +" → "Web Service"
2. **Select Repository**: Choose your food AI API repository
3. **Configure Service**:
   - **Name**: `food-ai-api` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free

### 4. Set Environment Variables
In your Render service dashboard, go to "Environment" tab and add:
- **Key**: `GOOGLE_API_KEY`
- **Value**: Your Google API key for Gemini Vision AI

### 5. Deploy
Click "Create Web Service" and wait for deployment (5-10 minutes).

## Important Notes

### Model Files
Your model files (~1.05 GB) will be uploaded during deployment. This is within Render's limits.

### Sleep Behavior
- **Free tier**: Service sleeps after 15 minutes of inactivity
- **First request**: May take 30-60 seconds to wake up
- **Subsequent requests**: Normal response times
- **Upgrade**: $7/month for always-on service

### Monitoring
- **Logs**: Available in Render dashboard
- **Health checks**: Automatic monitoring
- **Custom domain**: Can add your own domain

## Testing Your Deployment

Once deployed, test your endpoints:
```bash
# Health check
curl https://your-app-name.onrender.com/health

# Test food analysis
curl -X POST https://your-app-name.onrender.com/analyze \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_food_image.jpg"
```

## Troubleshooting

### Common Issues:
1. **Build fails**: Check `requirements.txt` for compatibility
2. **Model loading errors**: Ensure model files are in `models/` directory
3. **API key errors**: Verify `GOOGLE_API_KEY` environment variable
4. **Memory issues**: Free tier has 512MB RAM limit

### Performance Tips:
1. **Model optimization**: Consider using smaller models for faster loading
2. **Caching**: Implement response caching for repeated requests
3. **Async processing**: Use background tasks for heavy operations

## Cost Comparison

| Platform | Free Tier | Paid Plans | Best For |
|----------|-----------|------------|----------|
| **Render** | 750 hours/month | $7/month always-on | Best free option |
| Railway | 30-day trial | $5/month | Good paid option |
| Fly.io | 3 VMs, 3GB storage | Pay-per-use | Global deployment |
| Heroku | Discontinued | $5/month | Easy deployment |
| Google Cloud Run | 2M requests/month | Pay-per-use | Serverless |

## Next Steps
1. Deploy to Render using this guide
2. Test all endpoints thoroughly
3. Consider upgrading to paid plan if you need always-on service
4. Set up monitoring and alerts
5. Add custom domain if needed

Your Food AI API will be live and accessible worldwide! 🚀