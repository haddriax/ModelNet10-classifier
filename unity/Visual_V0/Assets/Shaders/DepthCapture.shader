Shader "Custom/DepthCapture"
{
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct v2f {
                float4 pos : SV_POSITION;
                float depth : TEXCOORD0;
            };

            v2f vert(appdata_base v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.depth = -UnityObjectToViewPos(v.vertex).z;
                return o;
            }

            float4 frag(v2f i) : SV_Target
            {
                // Normalise entre 0 et 1 par rapport au far clip
                float d = saturate(i.depth / _ProjectionParams.z);
                // Grayscale simple sur les 3 canaux
                return float4(d, d, d, 1.0);
            }
            ENDCG
        }
    }
}