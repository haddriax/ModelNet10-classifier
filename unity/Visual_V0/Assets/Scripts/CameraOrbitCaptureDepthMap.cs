using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Newtonsoft.Json;

public class CameraOrbitCaptureDepthMap : MonoBehaviour
{
    private static string OBJ_PATH = "Models/models_prefabs";
    private GameObject[] modelsList;

    [Header("Paramètres orbite")]
    public int numCameras = 6;
    public int numElevations = 2;
    public float orbitRadius = 3.0f;

    [Header("Caméra")]
    public int imageWidth = 640;
    public int imageHeight = 480;
    public float cameraFOV = 60f;

    [Header("Export")]
    public string outputBasePath = "Assets/ModelsDatasetOutputLessViews";

    private Camera captureCamera;
    private Shader depthShader;

    void Start()
    {
        modelsList = Resources.LoadAll<GameObject>(OBJ_PATH);

        // Shader de capture de profondeur
        depthShader = Shader.Find("Custom/DepthCapture");
        if (depthShader == null)
            Debug.LogError("Shader 'Custom/DepthCapture' introuvable ! Vérifie Assets/Shaders/DepthCapture.shader");

        GameObject camGO = new GameObject("CaptureCamera");
        captureCamera = camGO.AddComponent<Camera>();
        captureCamera.fieldOfView = cameraFOV;
        captureCamera.clearFlags = CameraClearFlags.SolidColor;
        captureCamera.backgroundColor = Color.black;
        captureCamera.nearClipPlane = 0.01f;
        captureCamera.farClipPlane = 100f;

        StartCoroutine(PlaceObjectAndTakeScreenshots());
    }

    IEnumerator PlaceObjectAndTakeScreenshots()
    {
        foreach (GameObject model in modelsList)
        {
            GameObject obj = Instantiate(model, Vector3.zero, Quaternion.identity);
            obj.name = model.name; // supprime "(Clone)"

            obj.transform.rotation = Quaternion.Euler(-90f, 0f, 0f);

            string objFolder = Path.Combine(outputBasePath, obj.name);
            Directory.CreateDirectory(objFolder);

            Bounds bounds = GetBounds(obj);
            Vector3 center = bounds.center;
            float radius = Mathf.Max(bounds.extents.magnitude * 2f, orbitRadius);

            List<CameraInfo> cameraInfos = new List<CameraInfo>();
            int frameIndex = 0;

            float[] elevations = new float[numElevations];
            for (int e = 0; e < numElevations; e++)
                //elevations[e] = Mathf.Lerp(-20f, 40f, (float)e / (numElevations - 1));
                elevations[e] = Mathf.Lerp(0f, 30f, (float)e / (numElevations - 1));

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
                    string depthName = $"depth_{frameIndex:D4}.png";

                    SaveCameraRender(Path.Combine(objFolder, imageName));
                    SaveDepth(Path.Combine(objFolder, depthName));

                    cameraInfos.Add(new CameraInfo
                    {
                        frame_id = frameIndex,
                        image_name = imageName,
                        depth_name = depthName,
                        position = Vec3ToArray(captureCamera.transform.position),
                        rotation = MatrixToArray(Matrix4x4.Rotate(captureCamera.transform.rotation)),
                        projection_matrix = MatrixToArray(captureCamera.projectionMatrix),
                        view_matrix = MatrixToArray(captureCamera.worldToCameraMatrix),
                        fov = cameraFOV,
                        width = imageWidth,
                        height = imageHeight,
                        near_clip = captureCamera.nearClipPlane,
                        far_clip = captureCamera.farClipPlane,
                        cam_to_world = MatrixToArray(captureCamera.transform.localToWorldMatrix)
                    });

                    frameIndex++;
                }
            }

            string json = JsonConvert.SerializeObject(cameraInfos, Formatting.Indented);
            File.WriteAllText(Path.Combine(objFolder, "cameras.json"), json);

            Debug.Log($"[{obj.name}] {frameIndex} frames capturées → {objFolder}");
            Destroy(obj);
            yield return null;
        }

        Debug.Log("=== End of the process ===");
#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
#endif
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

    void SaveDepth(string path)
    {
        if (depthShader == null) return;

        RenderTexture rt = new RenderTexture(imageWidth, imageHeight, 24);
        captureCamera.targetTexture = rt;
        captureCamera.RenderWithShader(depthShader, "");

        RenderTexture.active = rt;
        Texture2D tex = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        tex.Apply();
        File.WriteAllBytes(path, tex.EncodeToPNG());

        captureCamera.targetTexture = null;
        RenderTexture.active = null;
        Destroy(rt);
        Destroy(tex);
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
        public string depth_name;
        public float[] position;
        public float[] rotation;
        public float[] projection_matrix;
        public float[] view_matrix;
        public float fov;
        public int width, height;
        public float near_clip;
        public float far_clip;
        public float[] cam_to_world;
    }
}