using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using Newtonsoft.Json;

public class ScreenshotCapturePlceCamera : MonoBehaviour
{
    private static string OBJ_PATH = "Models/models_prefabs";
    private GameObject[] modelsList;

    [Header("Paramètres stéréo")]
    public float baseline = 10f;        
    public float distanceToObject = 30f; 
    public float cameraFOV = 60f;

    [Header("Résolution")]
    public int imageWidth = 640;
    public int imageHeight = 480;

    [Header("Export")]
    public string outputBasePath = "Assets/ScreenShots";

    private Camera cameraLeft;
    private Camera cameraRight;

    void Start()
    {
        modelsList = Resources.LoadAll<GameObject>(OBJ_PATH);

        GameObject goL = new GameObject("CameraLeft");
        cameraLeft = goL.AddComponent<Camera>();
        cameraLeft.fieldOfView = cameraFOV;
        cameraLeft.clearFlags = CameraClearFlags.SolidColor;
        cameraLeft.backgroundColor = Color.black;

        GameObject goR = new GameObject("CameraRight");
        cameraRight = goR.AddComponent<Camera>();
        cameraRight.fieldOfView = cameraFOV;
        cameraRight.clearFlags = CameraClearFlags.SolidColor;
        cameraRight.backgroundColor = Color.black;

        StartCoroutine(PlaceObjectAndTakeScreenshots());
    }

    void PlaceStereocameras(Vector3 target)
    {
        Vector3 posLeft  = target + new Vector3(-baseline / 2f, 0f, -distanceToObject);
        Vector3 posRight = target + new Vector3( baseline / 2f, 0f, -distanceToObject);

        cameraLeft.transform.position  = posLeft;
        cameraRight.transform.position = posRight;

        cameraLeft.transform.LookAt(target);
        cameraRight.transform.LookAt(target);
    }

    IEnumerator PlaceObjectAndTakeScreenshots()
    {
        foreach (GameObject model in modelsList)
        {
            GameObject obj = Instantiate(model, Vector3.zero, Quaternion.identity);
            obj.name = model.name;

            obj.transform.rotation = Quaternion.Euler(-90f, 0f, 0f);

            string objFolder = Path.Combine(outputBasePath, obj.name);
            Directory.CreateDirectory(objFolder);

            Bounds bounds = GetBounds(obj);
            Vector3 center = bounds.center;

            PlaceStereocameras(center);

            yield return new WaitForEndOfFrame();

            string pathLeft  = Path.Combine(objFolder, "left.png");
            string pathRight = Path.Combine(objFolder, "right.png");
            SaveRender(cameraLeft,  pathLeft);
            SaveRender(cameraRight, pathRight);

       
            var camInfos = new List<CameraInfo>
            {
                BuildCameraInfo("left",  cameraLeft),
                BuildCameraInfo("right", cameraRight)
            };
            string json = JsonConvert.SerializeObject(camInfos, Formatting.Indented);
            File.WriteAllText(Path.Combine(objFolder, "cameras.json"), json);

            Destroy(obj);
            yield return null;
        }

        Debug.Log("=== End of the process ===");
#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
#endif
    }

    void SaveRender(Camera cam, string path)
    {
        RenderTexture rt = new RenderTexture(imageWidth, imageHeight, 24);
        cam.targetTexture = rt;
        cam.Render();

        RenderTexture.active = rt;
        Texture2D img = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        img.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        img.Apply();
        File.WriteAllBytes(path, img.EncodeToPNG());

        cam.targetTexture = null;
        RenderTexture.active = null;
        Destroy(rt);
        Destroy(img);
    }

    CameraInfo BuildCameraInfo(string name, Camera cam)
    {
        return new CameraInfo
        {
            name = name,
            position = Vec3ToArray(cam.transform.position),
            rotation = MatrixToArray(Matrix4x4.Rotate(cam.transform.rotation)),
            view_matrix = MatrixToArray(cam.worldToCameraMatrix),
            projection_matrix = MatrixToArray(cam.projectionMatrix),
            cam_to_world = MatrixToArray(cam.transform.localToWorldMatrix),
            fov = cameraFOV,
            width = imageWidth,
            height = imageHeight,
            near_clip = cam.nearClipPlane,
            far_clip = cam.farClipPlane,
            baseline = baseline
        };
    }

    Bounds GetBounds(GameObject obj)
    {
        Renderer[] renderers = obj.GetComponentsInChildren<Renderer>();
        if (renderers.Length == 0) return new Bounds(obj.transform.position, Vector3.one);
        Bounds b = renderers[0].bounds;
        foreach (var r in renderers) b.Encapsulate(r.bounds);
        return b;
    }

    float[] Vec3ToArray(Vector3 v) => new float[] { v.x, v.y, v.z };

    float[] MatrixToArray(Matrix4x4 m)
    {
        float[] arr = new float[16];
        for (int i = 0; i < 16; i++) arr[i] = m[i];
        return arr;
    }

    [System.Serializable]
    class CameraInfo
    {
        public string  name;
        public float[] position;
        public float[] rotation;
        public float[] view_matrix;
        public float[] projection_matrix;
        public float[] cam_to_world;
        public float   fov;
        public int     width, height;
        public float   near_clip, far_clip;
        public float   baseline;
    }
}