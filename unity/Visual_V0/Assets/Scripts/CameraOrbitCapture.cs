using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Newtonsoft.Json;

public class CameraOrbitCapture : MonoBehaviour
{
    private static string OBJ_PATH = "Models/models_prefabs";
    private GameObject[] modelsList;

    public int numCameras = 12;        
    public int numElevations = 3;      
    public float orbitRadius = 3.0f;

    public int imageWidth = 640;
    public int imageHeight = 480;
    public float cameraFOV = 60f;

    public string outputBasePath = "Assets/ModelsDatasetOutput";

    private Camera captureCamera;

    void Start()
    {
        modelsList = Resources.LoadAll<GameObject>(OBJ_PATH);

        GameObject camGO = new GameObject("CaptureCamera");
        captureCamera = camGO.AddComponent<Camera>();
        captureCamera.fieldOfView = cameraFOV;
        captureCamera.clearFlags = CameraClearFlags.SolidColor;
        captureCamera.backgroundColor = Color.black;
 
        StartCoroutine(PlaceObjectAndTakeScreenshots());
    }

    IEnumerator PlaceObjectAndTakeScreenshots()
    {        
        foreach (GameObject model in modelsList)
        {
            GameObject obj = Instantiate(model, Vector3.zero, Quaternion.identity);

            string objName = obj.name;
            string objFolder = Path.Combine(outputBasePath, objName);
            Directory.CreateDirectory(objFolder);

            Bounds bounds = GetBounds(obj);
            Vector3 center = bounds.center;

            float radius = Mathf.Max(bounds.extents.magnitude * 2f, orbitRadius);

            List<CameraInfo> cameraInfos = new List<CameraInfo>();
            int frameIndex = 0;

            float[] elevations = new float[numElevations];
            for (int e = 0; e < numElevations; e++)
                elevations[e] = Mathf.Lerp(-20f, 40f, (float)e / (numElevations - 1));

            foreach (float elev in elevations)
            {
                for (int i = 0; i < numCameras; i++)
                {
                    float azimuth = (360f / numCameras) * i;

                    float azRad = azimuth * Mathf.Deg2Rad;
                    float elRad = elev * Mathf.Deg2Rad;
                    Vector3 offset = new Vector3(
                        radius * Mathf.Cos(elRad) * Mathf.Sin(azRad),
                        radius * Mathf.Sin(elRad),
                        radius * Mathf.Cos(elRad) * Mathf.Cos(azRad)
                    );

                    captureCamera.transform.position = center + offset;
                    captureCamera.transform.LookAt(center);

                    yield return new WaitForEndOfFrame();

                    string imageName = $"frame_{frameIndex:D4}.png";
                    string imagePath = Path.Combine(objFolder, imageName);
                    SaveCameraRender(imagePath);

                    cameraInfos.Add(new CameraInfo
                    {
                        frame_id = frameIndex,
                        image_name = imageName,
                        position = Vec3ToArray(captureCamera.transform.position),
                        rotation = MatrixToArray(Matrix4x4.Rotate(captureCamera.transform.rotation)),
                        projection_matrix = MatrixToArray(captureCamera.projectionMatrix),
                        view_matrix = MatrixToArray(captureCamera.worldToCameraMatrix),
                        fov = cameraFOV,
                        width = imageWidth,
                        height = imageHeight
                    });

                    frameIndex++;
                }
            }

            string json = JsonConvert.SerializeObject(cameraInfos, Formatting.Indented);
            File.WriteAllText(Path.Combine(objFolder, "cameras.json"), json);

            Destroy(obj);
            yield return null;
        }
        Debug.Log("End of the process");
    }

    void SaveCameraRender(string path)
    {
        RenderTexture rt = new RenderTexture(imageWidth, imageHeight, 24);
        captureCamera.targetTexture = rt;
        captureCamera.Render();

        RenderTexture.active = rt;
        Texture2D img = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        img.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        img.Apply();

        File.WriteAllBytes(path, img.EncodeToPNG());

        captureCamera.targetTexture = null;
        RenderTexture.active = null;
        Destroy(rt);
        Destroy(img);
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
        public int frame_id;
        public string image_name;
        public float[] position;
        public float[] rotation;
        public float[] projection_matrix;
        public float[] view_matrix;
        public float fov;
        public int width, height;
    }
}