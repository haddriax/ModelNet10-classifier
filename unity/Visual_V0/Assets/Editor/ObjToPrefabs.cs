using UnityEngine;
using UnityEditor;
using System.IO;

public class ObjToPrefabs
{
    [MenuItem("Tools/Convert OBJ to Prefabs")]
    static void ConvertAllObjToPrefab()
    {
        string objPath = "Assets/Resources/Models/models_obj";
        string prefabPath = "Assets/Resources/Models/models_prefabs";

        string[] objFiles = Directory.GetFiles(objPath, "*.obj");

        foreach (string objFile in objFiles)
        {
            string assetPath = objFile.Replace("\\", "/");
            GameObject obj = AssetDatabase.LoadAssetAtPath<GameObject>(assetPath);

            if (obj != null)
            {
                string prefabName = Path.GetFileNameWithoutExtension(assetPath);
                string savePath = prefabPath + "/" + prefabName + ".prefab";

                PrefabUtility.SaveAsPrefabAsset(obj, savePath);
            }
        }

        AssetDatabase.Refresh();
        Debug.Log("End of conversion");
    }
}