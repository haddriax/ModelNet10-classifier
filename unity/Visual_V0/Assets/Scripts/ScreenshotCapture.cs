using System.Collections.Generic;
using TMPro;
using UnityEngine;
using System.IO;
using System.Collections;

public class ScreenshotCapture : MonoBehaviour
{
    public GameObject cameraQ;
    public GameObject cameraP;

    private static string OBJ_PATH = "Models/models_prefabs";

    private GameObject[] modelsList;

    void Start()
    {
        modelsList = Resources.LoadAll<GameObject>(OBJ_PATH);
        StartCoroutine(PlaceObjectAndTakeScreenshots());
    }


    //void Update()
    //{
    //    Take screeshot manually
    //    if (Input.GetKeyDown(KeyCode.Q))
    //    {
    //        TakeScreenshot(cameraQ);
    //    }
    //    if (Input.GetKeyDown(KeyCode.P))
    //    {
    //        TakeScreenshot(cameraP);
    //    }
    //}

    IEnumerator PlaceObjectAndTakeScreenshots()
    {        
        foreach (GameObject model in modelsList)
        {
            Debug.Log("Start...");
            GameObject obj = Instantiate(model, Vector3.zero, Quaternion.identity);

            yield return new WaitForEndOfFrame();
            TakeScreenshot(cameraQ, model.name);
            Debug.Log("1st screenshot taken...");

            yield return new WaitForEndOfFrame();
            TakeScreenshot(cameraP, model.name);

            Destroy(obj);
            yield return null;
        }
        Debug.Log("End of the process");
    }

    void TakeScreenshot(GameObject camObj, string objName)
    {
        foreach (Camera cam in Camera.allCameras)
            cam.enabled = false;
        
        string fileName = "Assets/ScreenShots/" + camObj.name + "_" + objName + ".png";

        if (camObj != null)
        {
            Camera cam = camObj.GetComponent<Camera>();
            cam.enabled = true;
            ScreenCapture.CaptureScreenshot(fileName);
        }

        foreach (Camera cam in Camera.allCameras)
        cam.enabled = true;
    }
}