using UnityEngine;
using UnityEditor;
using Unity.Profiling;
using UnityEngine.SceneManagement;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.IO;
using System.Drawing;
// Import ML agent libraries
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
// To compute the SSIM
// using AForge.Imaging;
// using AForge.Imaging.Filters;
// using System.Drawing;

// Compute Structural Similarity Index (SSIM) image quality metrics                                                                                 
// See http://www.ece.uwaterloo.ca/~z70wang/research/ssim/                                                                                          
// C# implementation Copyright Chris Lomont 2009-2011                                                                                               
// Send fixes and comments to WWW.LOMONT.ORG                                                                                                        
                                                                                                                                                    
/* History:                                                                                                                                         
 * June 2011                                                                                                                                        
 * 0.91 - Fix to Bitmap creation to prevent locking file handles                                                                                    
 *                                                                                                                                                  
 * Sept 2009                                                                                                                                        
 * 0.9  - Initial Release                                                                                                                           
 *                                                                                                                                                  
 *                                                                                                                                                  
 */                                                                                                                                               
                                                                                                                                    
class SSIM                                                                                                                                      
{                                                                                                                                           
    /// <summary>                                                                                                                               
    /// The version of the SSIM code                                                                                                            
    /// </summary>                                                                                                                              
    internal static string Version { get { return "0.91";} }                                                                                    
                                                                                                                                                
    /// <summary>                                                                                                                               
    /// Compute from two linear ushort grayscale images with given size and bitdepth                                                            
    /// </summary>                                                                                                                              
    /// <param name="img1">Image 1 data</param>                                                                                                 
    /// <param name="img2">Image 2 data</param>                                                                                                 
    /// <param name="w">width</param>                                                                                                           
    /// <param name="h">height</param>                                                                                                          
    /// <param name="depth">Bit depth (1-16)</param>                                                                                            
    /// <returns></returns>                                                                                                                     
    internal double Index(ushort[] img1, ushort[] img2, int w, int h, int depth)                                                                
        {                                                                                                                                       
        L = (1 << depth) - 1;                                                                                                                   
        return ComputeSSIM(ConvertLinear(img1, w, h), ConvertLinear(img2, w, h));                                                               
        }                                                                                                                                       
                                                                                                                                                
    /// <summary>                                                                                                                               
    /// Take two System.Drawing.Bitmaps                                                                                                         
    /// </summary>                                                                                                                              
    /// <param name="img1"></param>                                                                                                             
    /// <param name="img2"></param>                                                                                                             
    /// <returns></returns>                                                                                                                     
    internal double Index(Bitmap img1, Bitmap img2)                                                                                             
        {                                                                                                                                       
        L = 255; // todo - this assumes 8 bit, but color conversion later is always 8 bit, so ok?                                          
        return ComputeSSIM(ConvertBitmap(img1), ConvertBitmap(img2));                                                                           
        }                                                                                                                                       
                                                                                                                                                
    /// <summary>                                                                                                                               
    /// Take two filenames                                                                                                                      
    /// </summary>                                                                                                                              
    /// <param name="filename1"></param>                                                                                                        
    /// <param name="filename2"></param>                                                                                                        
    /// <returns></returns>                                                                                                                     
    internal double Index(string filename1, string filename2)                                                                                   
        {                                                     
        // Debug.Log("First filename: " + filename1 + ", Second filename: " + filename2);
        using (var b1 = new Bitmap(filename1)){
            // Debug.Log("First bitmap: " + b1);
            // Debug.Log(string.Format("{0}px by {1}px", b1.Width, b1.Height));                                                                                                  
            using (var b2 = new Bitmap(filename2)){
                // Debug.Log("Second bitmap: " + b2);
                // Debug.Log(string.Format("{0}px by {1}px", b2.Width, b2.Height));
                return Index(b1,b2);
            }
        }                                                                                                                                                                          
        // return 0;
        }                                                                                                                                       
                                                                                                                                                
    #region Implementation                                                                                                                      
                                                                                                                                                
    #region Locals                                                                                                                              
    // default settings, names from paper                                                                                                       
    internal double K1 = 0.01, K2 = 0.03;                                                                                                       
    internal double L = 255;                                                                                                                    
    readonly Grid window = Gaussian(11, 1.5);                                                                                                   
    #endregion                                                                                                                                  
                                                                                                                                                
    /// <summary>                                                                                                                               
    /// Compute the SSIM index of two same sized Grids                                                                                          
    /// </summary>                                                                                                                              
    /// <param name="img1">The first Grid</param>                                                                                               
    /// <param name="img2">The second Grid</param>                                                                                              
    /// <returns>SSIM index</returns>                                                                                                           
    double ComputeSSIM(Grid img1, Grid img2)                                                                                                    
        {                                                                                                                                       
        // uses notation from paper                                                                                                             
        // automatic downsampling                                                                                                               
        int f = (int)Math.Max(1, Math.Round(Math.Min(img1.width,img1.height) / 256.0));                                                         
        if (f > 1)                                                                                                                              
            { // downsampling by f                                                                                                              
                // use a simple low-pass filter and subsample by f                                                                                
            img1 = SubSample(img1, f);                                                                                                          
            img2 = SubSample(img2, f);                                                                                                          
            }                                                                                                                                   
                                                                                                                                                
        // normalize window - todo - do in window set {}                                                                                        
        double scale = 1.0/window.Total;                                                                                                        
        Grid.Op((i, j) => window[i, j] * scale, window);                                                                                        
                                                                                                                                                
        // image statistics                                                                                                                     
        var mu1 = Filter(img1, window);                                                                                                         
        var mu2 = Filter(img2, window);                                                                                                         
                                                                                                                                                
        var mu1mu2 = mu1 * mu2;                                                                                                                 
        var mu1SQ  = mu1 * mu1;                                                                                                                 
        var mu2SQ  = mu2 * mu2;                                                                                                                 
                                                                                                                                                
        var sigma12  = Filter(img1 * img2, window) - mu1mu2;                                                                                    
        var sigma1SQ = Filter(img1 * img1, window) - mu1SQ;                                                                                     
        var sigma2SQ = Filter(img2 * img2, window) - mu2SQ;                                                                                     
                                                                                                                                                
        // constants from the paper                                                                                                             
        double C1 = K1 * L; C1 *= C1;                                                                                                           
        double C2 = K2 * L; C2 *= C2;                                                                                                           
                                                                                                                                                
        Grid ssim_map = null;                                                                                                                   
        if ((C1 > 0) && (C2 > 0))                                                                                                               
            {                                                                                                                                   
            ssim_map = Grid.Op((i, j) =>                                                                                                        
                (2 * mu1mu2[i, j] + C1) * (2 * sigma12[i, j] + C2) /                                                                            
                (mu1SQ[i, j] + mu2SQ[i, j] + C1) / (sigma1SQ[i, j] + sigma2SQ[i, j] + C2),                                                      
                new Grid(mu1mu2.width, mu1mu2.height));                                                                                         
            }                                                                                                                                   
        else                                                                                                                                    
            {                                                                                                                                   
            var num1 = Linear(2, mu1mu2, C1);                                                                                                   
            var num2 = Linear(2, sigma12, C2);                                                                                                  
            var den1 = Linear(1, mu1SQ + mu2SQ, C1);                                                                                            
            var den2 = Linear(1, sigma1SQ + sigma2SQ, C2);                                                                                      
                                                                                                                                                
            var den = den1 * den2; // total denominator                                                                                         
            ssim_map = new Grid(mu1.width, mu1.height);                                                                                         
            for (int i = 0; i < ssim_map.width; ++i)                                                                                            
                for (int j = 0; j < ssim_map.height; ++j)                                                                                       
                    {                                                                                                                           
                    ssim_map[i, j] = 1;                                                                                                         
                    if (den[i, j] > 0)                                                                                                          
                        ssim_map[i, j] = num1[i, j] * num2[i, j] / (den1[i, j] * den2[i, j]);                                                   
                    else if ((den1[i, j] != 0) && (den2[i, j] == 0))                                                                            
                        ssim_map[i, j] = num1[i, j] / den1[i, j];                                                                               
                    }                                                                                                                           
            }                                                                                                                                   
                                                                                                                                                
        // average all values                                                                                                                   
        return ssim_map.Total / (ssim_map.width * ssim_map.height);                                                                             
        } // ComputeSSIM                                                                                                                        
                                                                                                                                                
                                                                                                                                                
    #region Grid                                                                                                                                
    /// <summary>                                                                                                                               
    /// Hold a grid of doubles as an array with appropriate operators                                                                           
    /// </summary>                                                                                                                              
    class Grid                                                                                                                                  
        {                                                                                                                                       
        double[,] data;                                                                                                                         
        internal int width, height;                                                                                                             
        internal Grid(int w, int h)                                                                                                             
            {                                                                                                                                   
            data = new double[w, h];                                                                                                            
            width = w;                                                                                                                          
            height = h;                                                                                                                         
            }                                                                                                                                   
                                                                                                                                                
        /// <summary>                                                                                                                           
        /// Indexer to read the i,j item                                                                                                        
        /// </summary>                                                                                                                          
        /// <param name="i"></param>                                                                                                            
        /// <param name="j"></param>                                                                                                            
        /// <returns></returns>                                                                                                                 
        internal double this[int i, int j]                                                                                                      
            {                                                                                                                                   
            get { return data[i, j]; }                                                                                                          
            set { data[i, j] = value; }                                                                                                         
            }                                                                                                                                   
                                                                                                                                                
        /// <summary>                                                                                                                           
        /// Get the summed value from the Grid                                                                                                  
        /// </summary>                                                                                                                          
        internal double Total                                                                                                                   
            {                                                                                                                                   
            get                                                                                                                                 
                {                                                                                                                               
                double s = 0;                                                                                                                   
                foreach (var d in data) s += d;                                                                                                 
                return s;                                                                                                                       
                }                                                                                                                               
            }                                                                                                                                   
                                                                                                                                                
        /// <summary>                                                                                                                           
        /// componentwise addition of Grids                                                                                                     
        /// </summary>                                                                                                                          
        /// <param name="a"></param>                                                                                                            
        /// <param name="b"></param>                                                                                                            
        /// <returns></returns>                                                                                                                 
        static public Grid operator+(Grid a, Grid b)                                                                                            
            {                                                                                                                                   
            return Op((i,j)=>a[i,j]+b[i,j],new Grid(a.width,a.height));                                                                         
            }                                                                                                                                   
                                                                                                                                                
        /// <summary>                                                                                                                           
        /// componentwise subtraction of Grids                                                                                                  
        /// </summary>                                                                                                                          
        /// <param name="a"></param>                                                                                                            
        /// <param name="b"></param>                                                                                                            
        /// <returns></returns>                                                                                                                 
        static public Grid operator-(Grid a, Grid b)                                                                                            
            {                                                                                                                                   
            return Op((i, j) => a[i, j] - b[i, j], new Grid(a.width, a.height));                                                                
            }                                                                                                                                   
                                                                                                                                                
        /// <summary>                                                                                                                           
        /// componentwise multiplication of Grids                                                                                               
        /// </summary>                                                                                                                          
        /// <param name="a"></param>                                                                                                            
        /// <param name="b"></param>                                                                                                            
        /// <returns></returns>                                                                                                                 
        static public Grid operator*(Grid a, Grid b)                                                                                            
            {                                                                                                                                   
            return Op((i, j) => a[i, j] * b[i, j], new Grid(a.width, a.height));                                                                
            }                                                                                                                                   
                                                                                                                                                
        /// <summary>                                                                                                                           
        /// componentwise division of Grids                                                                                                     
        /// </summary>                                                                                                                          
        /// <param name="a"></param>                                                                                                            
        /// <param name="b"></param>                                                                                                            
        /// <returns></returns>                                                                                                                 
        static public Grid operator/(Grid a, Grid b)                                                                                            
            {                                                                                                                                   
            return Op((i, j) => a[i, j] / b[i, j], new Grid(a.width, a.height));                                                                
            }                                                                                                                                   
                                                                                                                                                
        /// <summary>                                                                                                                           
        /// Generic function maps (i,j) onto the given grid                                                                                     
        /// </summary>                                                                                                                          
        /// <param name="f"></param>                                                                                                            
        /// <param name="a"></param>                                                                                                            
        /// <returns></returns>                                                                                                                 
        static internal Grid Op(Func<int,int,double> f,Grid g)                                                                                  
            {                                                                                                                                   
            int w = g.width, h = g.height;                                                                                                      
            for (int i = 0; i < w; ++i)                                                                                                         
                for (int j = 0; j < h; ++j)                                                                                                     
                    g[i, j] = f(i,j);                                                                                                           
            return g;                                                                                                                           
            }                                                                                                                                   
                                                                                                                                                
        } // class Grid                                                                                                                         
    #endregion //Grid                                                                                                                           
                                                                                                                                                
                                                                                                                                                
    /// <summary>                                                                                                                               
    /// Create a gaussian window of the given size and standard deviation                                                                       
    /// </summary>                                                                                                                              
    /// <param name="size">Odd number</param>                                                                                                   
    /// <param name="sigma">Gaussian std deviation</param>                                                                                      
    /// <returns></returns>                                                                                                                     
    static Grid Gaussian(int size, double sigma)                                                                                                
        {                                                                                                                                       
        var filter = new Grid(size, size);                                                                                                      
        double s2 = sigma * sigma, c = (size-1)/2.0, dx, dy;                                                                                    
                                                                                                                                                
        Grid.Op((i, j) =>                                                                                                                       
            {                                                                                                                                   
                dx = i - c;                                                                                                                     
                dy = j - c;                                                                                                                     
                return Math.Exp(-(dx * dx + dy * dy) / (2 * s2));                                                                               
            },                                                                                                                                  
            filter);                                                                                                                            
        var scale = 1.0/filter.Total;                                                                                                           
        Grid.Op((i, j) => filter[i, j] * scale, filter);                                                                                        
        return filter;                                                                                                                          
        }                                                                                                                                       
                                                                                                                                                
    /// <summary>                                                                                                                               
    /// subsample a grid by step size, averaging each box into the result value                                                                 
    /// </summary>                                                                                                                              
    /// <returns></returns>                                                                                                                     
    static Grid SubSample(Grid img, int skip)                                                                                                   
        {                                                                                                                                       
        int w = img.width;                                                                                                                      
        int h = img.height;                                                                                                                     
        double scale = 1.0 / (skip * skip);                                                                                                     
        var ans = new Grid(w / skip, h / skip);                                                                                                 
        for (int i = 0; i < w-skip ; i+=skip)                                                                                                   
            for (int j = 0; j < h-skip ; j+=skip)                                                                                               
                {                                                                                                                               
                double sum = 0;                                                                                                                 
                for (int x = i; x < i + skip; ++x)                                                                                              
                    for (int y = j; y < j+ skip; ++y)                                                                                           
                        sum += img[x, y];                                                                                                       
                ans[i/skip, j/skip] = sum * scale;                                                                                              
                }                                                                                                                               
        return ans;                                                                                                                             
        }                                                                                                                                       
                                                                                                                                                
    /// <summary>                                                                                                                               
    /// Apply filter, return only center part.                                                                                                  
    /// C = Filter(A,B) should be same as matlab filter2( ,'valid')                                                                             
    /// </summary>                                                                                                                              
    /// <returns></returns>                                                                                                                     
    static Grid Filter(Grid a, Grid b)                                                                                                          
        {                                                                                                                                       
#if false                                                                                                                                           
        int ax = a.width, ay = a.height;                                                                                                        
        int bx = b.width, by = b.height;                                                                                                        
        int bcx = (bx + 1) / 2, bcy = (by + 1) / 2; // center position                                                                          
        var c = new Grid(ax - bx + 1, ay - by + 1);                                                                                             
        for (int i = bx - bcx + 1; i < ax - bx; ++i)                                                                                            
            for (int j = by - bcy + 1; j < ay - by; ++j)                                                                                        
                {                                                                                                                               
                double sum = 0;                                                                                                                 
                for (int x = bcx - bx + 1 + i; x < 1 + i + bcx; ++x)                                                                            
                    for (int y = bcy - by + 1 + j; y < 1 + j + bcy; ++y)                                                                        
                        sum += a[x, y] * b[bx - bcx - 1 - i + x, by - bcy - 1 - j + y];                                                         
                c[i - bcx, j - bcy] = sum;                                                                                                      
                }                                                                                                                               
        return c;                                                                                                                               
#else                                                                                                                                               
        // todo - check and clean this                                                                                                          
        int ax = a.width, ay = a.height;                                                                                                        
        int bx = b.width, by = b.height;                                                                                                        
        int bcx = (bx + 1) / 2, bcy = (by + 1) / 2; // center position                                                                          
        var c = new Grid(ax - bx + 1, ay - by + 1);                                                                                             
        for (int i = bx - bcx + 1; i < ax - bx; ++i)                                                                                            
            for (int j = by - bcy + 1; j < ay - by; ++j)                                                                                        
                {                                                                                                                               
                double sum = 0;                                                                                                                 
                for (int x = bcx - bx + 1 + i; x < 1 + i + bcx; ++x)                                                                            
                    for (int y = bcy - by + 1 + j; y < 1 + j + bcy; ++y)                                                                        
                        sum += a[x, y] * b[bx - bcx - 1 - i + x, by - bcy - 1 - j + y];                                                         
                c[i - bcx, j - bcy] = sum;                                                                                                      
                }                                                                                                                               
        return c;                                                                                                                               
#endif                                                                                                                                              
        }                                                                                                                                       
                                                                                                                                                
    /// <summary>                                                                                                                               
    /// componentwise s*a[i,j]+c->a[i,j]                                                                                                        
    /// </summary>                                                                                                                              
    /// <param name="s"></param>                                                                                                                
    /// <param name="a"></param>                                                                                                                
    /// <param name="c"></param>                                                                                                                
    /// <returns></returns>                                                                                                                     
    static Grid Linear(double s, Grid a, double c)                                                                                              
        {                                                                                                                                       
        return Grid.Op((i, j) => s * a[i, j] + c, new Grid(a.width,a.height));                                                                  
        }                                                                                                                                       
                                                                                                                                                
    #region Conversion                                                                                                                          
    /// <summary>                                                                                                                               
    /// convert image from 1D ushort to Grid                                                                                                    
    /// </summary>                                                                                                                              
    /// <param name="img"></param>                                                                                                              
    /// <param name="w"></param>                                                                                                                
    /// <param name="h"></param>                                                                                                                
    /// <returns></returns>                                                                                                                     
    static Grid ConvertLinear(ushort[] img, int w, int h)                                                                                       
        {                                                                                                                                       
        return Grid.Op((i,j)=>img[i+j*w],new Grid(w,h));                                                                                        
        }                                                                                                                                       
                                                                                                                                                
    /// <summary>                                                                                                                               
    /// Convert a Bitmap to a grayscale Grid                                                                                                    
    /// </summary>                                                                                                                              
    /// <returns></returns>                                                                                                                     
    static Grid ConvertBitmap(Bitmap bmp)                                                                                                       
        {                                                                                                                                       
        return Grid.Op((i, j) => { System.Drawing.Color c = bmp.GetPixel(i, j); return 0.3 * c.R + 0.59 * c.G + 0.11 * c.B; }, new Grid(bmp.Width,bmp.Height));
        }                                                                                                                                       
    #endregion // Conversion                                                                                                                    
                                                                                                                                                
    #endregion                                                                                                                                  
} // class SSIM

public class CameraSnapAllSSIM : Agent
{
    public float waitingInterval = 2f;
    public string cameraPath;
    public GameObject lodContainer;
    public int resWidth = 1024;
    public int resHeight = 768;
    public string folderScreenshots = "screenshots";
    // optimize for many screenshots will not destroy any objects so future screenshots will be fast
    public bool optimizeForManyScreenshots = true;
    public enum Format { RAW, JPG, PNG, PPM };
    public Format format = Format.PNG;

    public float fTotalTime = 2.0f;  // seconds
    // float fDist = 100.0f; //100 px
    // float fSpeed = fDist / fTotalTime;
    // public float speed = 2f;

    // private vars for screenshot
    private Rect rect;
    private RenderTexture renderTexture;
    private Texture2D screenShot;

    private double lastTime;
    private int lastFrameCount;
    private Vector3[] posSequence;
    private Vector3[] orienSequence;
    private Vector3 initCamPos;
    private int seqNum;
    private StreamWriter logFile;

    private Camera cam;
    private float previousQuality = 0f;
    // private int currentLod = 0;
    private int currentLod = -1;
    private bool running = true;
    private Agent agent;

    /** Beginning of Agent */

    EnvironmentParameters m_resetParams;

    public override void Initialize()
    {
        Start();

        agent = gameObject.GetComponent<Agent>();

        m_resetParams = Academy.Instance.EnvironmentParameters;
        SetResetParameters();
    }

    void Update()
    {
        double timeInterval = Time.realtimeSinceStartup - lastTime;
        // Debug.Log("timeInterval: " + timeInterval + "s");
        if (timeInterval > waitingInterval)
        {
            RequestDecision();
            
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Add SSIM quality, starting position, end position
        initCamPos = cam.transform.position;
        sensor.AddObservation(initCamPos);  // Starting Position
        sensor.AddObservation(posSequence[seqNum]);  // End Position
        sensor.AddObservation(previousQuality);  // Quality
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // Actions, size = 2
        int selectedLod = actionBuffers.DiscreteActions[0];
        // for (int i = 0; i < actionBuffers.DiscreteActions.Length; i++)
        // {
        //     if (actionBuffers.DiscreteActions[i] == 1)
        //     {
        //         selectedLod = i;
        //         break;
        //     }
        // }

        // if (selectedLod == -1)
        // {
        //     return;
        // }

        Debug.Log("Active LoD: " + currentLod);
        Debug.Log("Selected LoD: " + selectedLod);

        Debug.Log("Current sequence number:" + seqNum);
        
        foreach (Transform child in lodContainer.transform)
            child.gameObject.SetActive(false);

        // lodContainer.transform.GetChild(currentLod).gameObject.SetActive(false);
        
        lodContainer.transform.GetChild(selectedLod).gameObject.SetActive(true);

        currentLod = selectedLod;

        Update_Environment();

        // Rewards
        float camSpeed = Vector3.Distance(initCamPos, posSequence[seqNum]);
        float camDistance = Vector3.Distance(initCamPos, lodContainer.transform.position);
        // float distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition);
        float reward = (1 - Mathf.Abs((float) (1 - 0.5 * camSpeed - 0.5 * camDistance) - previousQuality));

        SetReward(reward);

        seqNum++;
        Debug.Log("Sequence Number updated to: " + seqNum);

        if (seqNum == posSequence.Length)
            NextSequence();

        if (!running)
            return;

        Debug.Log("Sequence Number check: " + seqNum);

        lastTime = Time.realtimeSinceStartup;
        lastFrameCount = Time.frameCount;
        MoveCam(seqNum);

        // EndEpisode();

        // Update the sequence number
        // seqNum++;
        // if (seqNum == posSequence.Length)
        //     NextSequence();

        // if (!running)
        //     return;

        // lastTime = Time.realtimeSinceStartup;
        // lastFrameCount = Time.frameCount;
        // MoveCam(seqNum);
    }

    public override void OnEpisodeBegin()
    {
        //Reset the parameters when the Agent is reset.
        SetResetParameters();
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        // if (Input.GetKeyDown("0"))
        if (Input.GetKey("0"))
        {
            discreteActionsOut[0] = 0;
            Debug.Log("Pressed 0");
        } else if (Input.GetKey("1"))
        {
            discreteActionsOut[0] = 1;
            Debug.Log("Pressed 1");
        } else if (Input.GetKey("2"))
        {
            discreteActionsOut[0] = 2;
            Debug.Log("Pressed 2");
        } else if (Input.GetKey("3"))
        {
            discreteActionsOut[0] = 3;
            Debug.Log("Pressed 3");
        }
    }

    public void SetResetParameters()
    {
        Debug.Log("Resetting parameters.");
        //Set the attributes of the ball by fetching the information from the academy
        seqNum = 0;
        currentLod = -1;
        initCamPos = posSequence[seqNum];  // Starting Position
        cam.transform.position = initCamPos;
        cam.transform.LookAt(lodContainer.transform.position - initCamPos);
        previousQuality = 0f;  // Quality
    }

    /** End of Agent */

    void Start()
    {

        Application.runInBackground = true;
        
        cam = this.GetComponent<Camera>();
        
        var lines = File.ReadAllLines(Application.dataPath + "/" + cameraPath);

        posSequence = new Vector3[lines.Length];
        orienSequence = new Vector3[lines.Length];
        string line;
        for (int i = 0; i < lines.Length; i++)
        {
            line = lines[i];
            // Very important to consider dots "." as decimal separators
            System.Globalization.CultureInfo culture = System.Globalization.CultureInfo.GetCultureInfo("en-US");
            // float[] floatData = Array.ConvertAll(line.Split(' '), float.Parse);
            List<float> floatData = line.Split(' ').Select(x => float.Parse(x, culture)).ToList();
            posSequence[i] = new Vector3(floatData[0], floatData[1], floatData[2]);
            orienSequence[i] = new Vector3(floatData[3], floatData[4], floatData[5]);
        }

        // log statistics
        // int triCount = UnityEditor.UnityStats.triangles;
        // int vertCount = UnityEditor.UnityStats.vertices;
        // int textCount = UnityEditor.UnityStats.renderTextureCount;
        int triCount = 0;
        int vertCount = 0;
        int textCount = 0;
        // double fps_c = (Time.frameCount - lastFrameCount) / timeInterval;
        double fps_c = 0;

        line = string.Format("seq_num: {0}; position: {1}; triangle_count: {2}; vertex_count: {3}; textures_count: {4}; fps: {5}",
                        seqNum, cam.transform.position, triCount, vertCount, textCount, fps_c);
        // logFile.WriteLine(line);

        string filename = TakeScreenshot();

        // print("Filename: " + filename);

        string groundTruth = GetImagePathFromLoD(0, seqNum);
        // print("Image path for current seqNumber (" + seqNum + ") is: " + GetImagePathFromLoD(currentLod, seqNum));
        print("Image path for current seqNumber (" + seqNum + ") and LoD (0) is: " + GetImagePathFromLoD(0, seqNum));

        // print("Computing SSIM for the current LoD");

        previousQuality = (float) compute_ssim(filename, groundTruth);

        print("The SSIM is: " + previousQuality);

        NextSequence();

        lastTime = Time.realtimeSinceStartup;
        lastFrameCount = Time.frameCount;

        MoveCam(seqNum);
    }

    private string GetDataPath()
    {
        // Init screenshot folder
        string path = Application.dataPath;
        if (Application.isEditor)
        {
            // put screenshots in folder above asset path so unity doesn't index the files
            var stringPath = path + "/..";
            path = Path.GetFullPath(stringPath);
        }
        return path;
    }

    private string GetImagePathFromLoD(int LoD, int seqNumber)
    {
        string lodString = lodContainer.transform.GetChild(LoD).gameObject.name;
        return string.Format("{0}/screenshots/{1}/{2}.{3}", GetDataPath(), lodString, seqNum, format.ToString().ToLower());
    }

    private void NextSequence()
    {
        // if (currentLod == lodContainer.transform.childCount - 1)
        // {
        //     #if UNITY_EDITOR
        //         UnityEditor.EditorApplication.isPlaying = false;
        //     #else
        //         Application.Quit();
        //     #endif
        //     running = false;
        //     return;
        // }
        if (currentLod == -1)
            currentLod = 0;
        else
        {
            logFile.Close();
            currentLod++;
        }

        foreach (Transform child in lodContainer.transform)
            child.gameObject.SetActive(false);

        GameObject lodObject = lodContainer.transform.GetChild(currentLod).gameObject;
        string lodName = lodObject.name;
        lodObject.SetActive(true);

        Debug.Log(lodName);

        string folderLog = string.Format("{0}/logData", GetDataPath());
        System.IO.Directory.CreateDirectory(folderLog);
        string fileTmst = string.Format("{0}/{1}.txt", folderLog, lodName);
        logFile = new StreamWriter(fileTmst, false);

        folderScreenshots = string.Format("{0}/screenshots/{1}", GetDataPath(), lodName);
        System.IO.Directory.CreateDirectory(folderScreenshots);

        // seqNum = 0;
    }

    public double compute_ssim(string testImagePath, string groundTruthImagePath)
    {
        // ComputeSSIM(ConvertBitmap(img1), ConvertBitmap(img2));
        SSIM ssim = new SSIM();

        return ssim.Index(testImagePath, groundTruthImagePath);
    }

    // void Update()
    // {
    //     double timeInterval = Time.realtimeSinceStartup - lastTime;
    //     if (timeInterval > waitingInterval)
    //     {
    //         // log statistics
    //         // int triCount = UnityEditor.UnityStats.triangles;
    //         // int vertCount = UnityEditor.UnityStats.vertices;
    //         // int textCount = UnityEditor.UnityStats.renderTextureCount;
    //         int triCount = 0;
    //         int vertCount = 0;
    //         int textCount = 0;
    //         double fps_c = (Time.frameCount - lastFrameCount) / timeInterval;

    //         string line = string.Format("seq_num: {0}; position: {1}; triangle_count: {2}; vertex_count: {3}; textures_count: {4}; fps: {5}",
    //                         seqNum, cam.transform.position, triCount, vertCount, textCount, fps_c);
    //         logFile.WriteLine(line);

    //         string filename = TakeScreenshot();

    //         print("Filename: " + filename);

    //         string groundTruth = GetImagePathFromLoD(0, seqNum);
    //         // print("Image path for current seqNumber (" + seqNum + ") is: " + GetImagePathFromLoD(currentLod, seqNum));
    //         print("Image path for current seqNumber (" + seqNum + ") and LoD (0) is: " + GetImagePathFromLoD(0, seqNum));

    //         print("Computing SSIM for the current LoD");

    //         previousQuality = (float) compute_ssim(filename, groundTruth);

    //         print("The SSIM is: " + previousQuality);

    //         // seqNum++;
    //         // if (seqNum == posSequence.Length)
    //         //     NextSequence();

    //         // if (!running)
    //         //     return;

    //         // lastTime = Time.realtimeSinceStartup;
    //         // lastFrameCount = Time.frameCount;
    //         // MoveCam(seqNum);
    //     }
    // }

    void Update_Environment()
    {
        // double timeInterval = Time.realtimeSinceStartup - lastTime;
        // if (timeInterval > waitingInterval)
        // if (condition)
        // {
            // log statistics
            // int triCount = UnityEditor.UnityStats.triangles;
            // int vertCount = UnityEditor.UnityStats.vertices;
            // int textCount = UnityEditor.UnityStats.renderTextureCount;
            int triCount = 0;
            int vertCount = 0;
            int textCount = 0;
            // double fps_c = (Time.frameCount - lastFrameCount) / timeInterval;
            double fps_c = 0;

            string line = string.Format("seq_num: {0}; position: {1}; triangle_count: {2}; vertex_count: {3}; textures_count: {4}; fps: {5}",
                            seqNum, cam.transform.position, triCount, vertCount, textCount, fps_c);
            logFile.WriteLine(line);

            string filename = TakeScreenshot();

            print("Filename: " + filename);

            string groundTruth = GetImagePathFromLoD(0, seqNum);
            // print("Image path for current seqNumber (" + seqNum + ") is: " + GetImagePathFromLoD(currentLod, seqNum));
            print("Image path for current seqNumber (" + seqNum + ") and LoD (0) is: " + GetImagePathFromLoD(0, seqNum));

            print("Computing SSIM for the current LoD");

            previousQuality = (float) compute_ssim(filename, groundTruth);

            print("The SSIM is: " + previousQuality);

            // seqNum++;
            // if (seqNum == posSequence.Length)
            //     NextSequence();

            // if (!running)
            //     return;

            // lastTime = Time.realtimeSinceStartup;
            // lastFrameCount = Time.frameCount;
            // MoveCam(seqNum);
        // }
    }

    private string TakeScreenshot()
    {
        // create screenshot objects if needed
        if (renderTexture == null)
        {
            // creates off-screen render texture that can rendered into
            rect = new Rect(0, 0, resWidth, resHeight);
            renderTexture = new RenderTexture(resWidth, resHeight, 24);
            screenShot = new Texture2D(resWidth, resHeight, TextureFormat.RGB24, false);
        }

        // get main camera and manually render scene into rt
        cam.targetTexture = renderTexture;
        cam.Render();
        // if (cam.transform.eulerAngles != Vector3.zero)
        // {
        //     cam.Render();
        // }

        // read pixels will read from the currently active render texture so make our offscreen 
        // render texture active and then read the pixels
        RenderTexture.active = renderTexture;
        screenShot.ReadPixels(rect, 0, 0);

        // reset active camera texture and render texture
        cam.targetTexture = null;
        RenderTexture.active = null;

        // get our unique filename
        string filename = UniqueFilename();

        // pull in our file header/data bytes for the specified image format (has to be done from main thread)
        byte[] fileHeader = null;
        byte[] fileData = null;

        if (format == Format.RAW)
        {
            fileData = screenShot.GetRawTextureData();
        }
        else if (format == Format.PNG)
        {
            fileData = screenShot.EncodeToPNG();
        }
        else if (format == Format.JPG)
        {
            fileData = screenShot.EncodeToJPG();
        }
        else // ppm
        {
            // create a file header for ppm formatted file
            string headerStr = string.Format("P6\n{0} {1}\n255\n", rect.width, rect.height);
            fileHeader = System.Text.Encoding.ASCII.GetBytes(headerStr);
            fileData = screenShot.GetRawTextureData();
        }

        // create new thread to save the image to file (only operation that can be done in background)
        // new System.Threading.Thread(() =>
        // {
        //     // create file and write optional header with image bytes
        //     var f = System.IO.File.Create(filename);
        //     if (fileHeader != null) f.Write(fileHeader, 0, fileHeader.Length);
        //     f.Write(fileData, 0, fileData.Length);
        //     f.Close();
        // }).Start();
        // create file and write optional header with image bytes

        var f = System.IO.File.Create(filename);
        if (fileHeader != null) f.Write(fileHeader, 0, fileHeader.Length);
        f.Write(fileData, 0, fileData.Length);
        f.Close();

        // cleanup if needed
        if (optimizeForManyScreenshots == false)
        {
            Destroy(renderTexture);
            renderTexture = null;
            screenShot = null;
        }

        return filename;
    }

    void OnApplicationQuit()
    {
        if (logFile != null)
        {
            logFile.Close();
        }
    }

    private string UniqueFilename()
    {
        return string.Format("{0}/{1}.{2}", folderScreenshots, seqNum, format.ToString().ToLower());
    }

    private IEnumerator Translate(Transform current, Vector3 target, float distance)
    {
        float step = distance / (fTotalTime);
        print("Step: " + step);
        while (Vector3.Distance(current.position, target) >= 0.01f)
        {
            current.position = Vector3.MoveTowards(current.position, target, step * Time.deltaTime);
            yield return null;
        }
    }

    private IEnumerator Rotate(Transform current, Vector3 futurePosition, Vector3 target)
    {
        // float stepAngle = angleToRotate / (fTotalTime);
        float currentTime = 0f;
        // Vector3 rotationPerSecond = (target - current.position) / fTotalTime;
        // print("Angle distance: " + Vector3.Distance(current.eulerAngles, target));
        while (Quaternion.Angle(current.rotation, Quaternion.Euler((target - futurePosition).x, (target - futurePosition).y, (target - futurePosition).z)) >= 0.01f)
        // while (Vector3.Distance(current.eulerAngles, target) >= 0.01f)
        {
            if (currentTime <= fTotalTime)
            {
                currentTime += Time.deltaTime;
                // current.eulerAngles = Vector3.Lerp(current.rotation.eulerAngles, target, currentTime / fTotalTime);
                // transform.Rotate(rotationPerSecond * Time.deltaTime);
                current.rotation = Quaternion.LookRotation(target - futurePosition, Vector3.up);
            }
            else
            {
                // current.eulerAngles = target;
                current.LookAt(target);
                currentTime = fTotalTime + 0.01f;
            }
            // current.RotateAround(target, Vector3.up, stepAngle * Time.deltaTime);
            // current.eulerAngles = Vector3.Lerp(current.rotation.eulerAngles, target, stepAngle * Time.deltaTime);
            // transform.rotation = Quaternion.Slerp(transform.rotation, otherObject.rotation, speed*Time.deltaTime);
            // print("Angle distance: " + Vector3.Distance(current.eulerAngles, target));
            yield return null;
        }
        current.LookAt(target);
        yield return null;
    }

    // private void MoveCam(int seqNum)
    // {
    //     // cam.transform.position = posSequence[seqNum];
    //     // Move our position a step closer to the target.
    //     // Vector3 nextOrientation = orienSequence[seqNum] - posSequence[seqNum];
    //     // Quaternion.Euler(nextOrientation.x, nextOrientation.y, nextOrientation.z)
    //     Vector3 targetPosition = lodContainer.transform.position;
    //     float angleToRotate = Quaternion.Angle(cam.transform.rotation, Quaternion.Euler(targetPosition.x, targetPosition.y, targetPosition.z));
    //     StartCoroutine(Rotate(cam.transform, posSequence[seqNum], targetPosition));
    //     // cam.transform.LookAt(lodContainer.transform);
    //     float distance = Vector3.Distance(posSequence[seqNum], cam.transform.position); // calculate distance to move
    //     print("Distance" + distance);
    //     StartCoroutine(Translate(cam.transform, posSequence[seqNum], distance));
    //     // cam.transform.position = Vector3.MoveTowards(cam.transform.position, posSequence[seqNum], step);
    //     // cam.transform.rotation = Quaternion.LookRotation(orienSequence[seqNum] - posSequence[seqNum], Vector3.up);
    //     // cam.transform.rotation = Quaternion.Lerp(cam.transform.rotation, Quaternion.Euler(nextOrientation.x, nextOrientation.y, nextOrientation.z), step);
    // }

    private void MoveCam(int seqNum)
    {
        Vector3 targetPosition = lodContainer.transform.position;
        cam.transform.LookAt(targetPosition);
        float distance = Vector3.Distance(posSequence[seqNum], cam.transform.position); // calculate distance to move
        cam.transform.position = posSequence[seqNum];
    }
}

