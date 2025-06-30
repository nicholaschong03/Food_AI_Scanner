# Render Deployment Checklist

## Pre-Deployment Checklist

### ✅ Code Preparation
- [ ] All Railway files removed (`railway.json`, `Procfile`, `DEPLOYMENT.md`)
- [ ] `render.yaml` created and configured
- [ ] `requirements.txt` optimized for Render
- [ ] `.gitignore` updated to exclude virtual environment
- [ ] README.md updated with Render deployment instructions

### ✅ GitHub Repository
- [ ] Code pushed to GitHub repository
- [ ] Repository is public (or you have Render Pro account)
- [ ] All model files included in repository
- [ ] No sensitive data in repository (API keys, etc.)

### ✅ API Keys Ready
- [ ] Google Gemini API key obtained
- [ ] Nutritionix API credentials (optional but recommended)
- [ ] Keys stored securely (not in code)

## Render Deployment Steps

### 1. Create Render Account
- [ ] Go to [render.com](https://render.com)
- [ ] Sign up with GitHub account
- [ ] Verify email address

### 2. Connect Repository
- [ ] Click "New +" → "Web Service"
- [ ] Connect GitHub account if not already connected
- [ ] Select your food AI API repository
- [ ] Choose the main branch

### 3. Configure Service
- [ ] **Name**: `food-ai-api` (or your preferred name)
- [ ] **Environment**: `Python 3`
- [ ] **Build Command**: `pip install -r requirements.txt`
- [ ] **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- [ ] **Plan**: Free

### 4. Set Environment Variables
- [ ] Go to "Environment" tab in service dashboard
- [ ] Add `GOOGLE_API_KEY` with your Gemini API key
- [ ] Add `NUTRITIONIX_APP_ID` and `NUTRITIONIX_APP_KEY` (optional)

### 5. Deploy
- [ ] Click "Create Web Service"
- [ ] Wait for build to complete (5-10 minutes)
- [ ] Note the deployment URL

## Post-Deployment Testing

### ✅ Health Check
- [ ] Test health endpoint: `https://your-app-name.onrender.com/health`
- [ ] Verify all models loaded successfully
- [ ] Check response time

### ✅ API Testing
- [ ] Test food detection endpoint
- [ ] Test food classification endpoint
- [ ] Test ingredient segmentation endpoint
- [ ] Test complete analysis endpoint
- [ ] Test nutrition analysis endpoint

### ✅ Performance Testing
- [ ] Test with different image sizes
- [ ] Verify response times are acceptable
- [ ] Check memory usage in logs
- [ ] Test sleep/wake behavior (free tier)

## Troubleshooting

### Common Issues
- [ ] **Build fails**: Check `requirements.txt` compatibility
- [ ] **Model loading errors**: Verify model files are in repository
- [ ] **API key errors**: Check environment variables
- [ ] **Memory issues**: Monitor logs for memory usage
- [ ] **Timeout errors**: Check image size limits

### Performance Optimization
- [ ] Consider model optimization for faster loading
- [ ] Implement response caching if needed
- [ ] Monitor API usage and costs
- [ ] Set up alerts for service health

## Next Steps

### After Successful Deployment
- [ ] Update documentation with your actual Render URL
- [ ] Test all endpoints thoroughly
- [ ] Set up monitoring and alerts
- [ ] Consider custom domain if needed
- [ ] Share your API with users

### Optional Upgrades
- [ ] Upgrade to paid plan for always-on service ($7/month)
- [ ] Add custom domain
- [ ] Set up CI/CD for automatic deployments
- [ ] Implement advanced monitoring

## Support Resources

- [Render Documentation](https://render.com/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Google Gemini API Docs](https://ai.google.dev/docs)
- [Nutritionix API Docs](https://www.nutritionix.com/business/api)

---

**Your Food AI API will be live and accessible worldwide! 🚀**