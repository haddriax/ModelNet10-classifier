using UnityEngine;

public class ScreenshotCapture : MonoBehaviour
{
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.A))
        {
            TakeScreenshot("CameraA", "Assets/image1.png");
        }
        if (Input.GetKeyDown(KeyCode.P))
        {
            TakeScreenshot("CameraP", "Assets/image2.png");
        }
    }

    void TakeScreenshot(string cameraName, string fileName)
    {
        foreach (Camera cam in Camera.allCameras)
            cam.enabled = false;

        GameObject camObj = GameObject.Find(cameraName);
        if (camObj != null)
        {
            Camera cam = camObj.GetComponent<Camera>();
            cam.enabled = true;
            ScreenCapture.CaptureScreenshot(fileName);
            Debug.Log("eeee");
        }
    }
}