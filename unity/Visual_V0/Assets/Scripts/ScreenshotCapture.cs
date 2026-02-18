using UnityEngine;

public class ScreenshotCapture : MonoBehaviour
{
    void Update()
    {
        SetCamera();
    }

    function SetCamera()
    {
        for (cam in Camera.allCameras)
		    cam.enabled = false;
	
        if (Input.GetKeyDown(KeyCode.A))
	        cam = GameObject.Find("???").GetComponent(Camera);

        if (Input.GetKeyDown(KeyCode.P))
            cam = GameObject.Find("???").GetComponent(Camera);
            
        cam.enabled = true;

        ScreenCapture.CaptureScreenshot("image1.png");
	    //TakeScreenshot(cam);
    }

    function TakeScreenshot(cam)
    {
        if (Input.GetKeyDown(KeyCode.A))
        {
            ScreenCapture.CaptureScreenshot("image1.png");
        }
        if (Input.GetKeyDown(KeyCode.P))
        {
            ScreenCapture.CaptureScreenshot("image2.png");
        }
    }
}