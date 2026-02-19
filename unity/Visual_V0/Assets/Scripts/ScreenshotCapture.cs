using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class ScreenshotCapture : MonoBehaviour
{
    public GameObject cameraQ;
    public GameObject cameraP;
    private GameObject[] objNames;    

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Q))
        {
            TakeScreenshot(cameraQ);
        }
        if (Input.GetKeyDown(KeyCode.P))
        {
            TakeScreenshot(cameraP);
        }
    }

    void TakeScreenshot(GameObject camObj)
    {
        foreach (Camera cam in Camera.allCameras)
            cam.enabled = false;

        objNames = GameObject.FindGameObjectsWithTag("Object");
        
        string fileName = "Assets/ScreenShots" + camObj.name + "_" + objNames[0].name + ".png";

        if (camObj != null)
        {
            Camera cam = camObj.GetComponent<Camera>();
            cam.enabled = true;
            ScreenCapture.CaptureScreenshot(fileName);
        }
    }
}