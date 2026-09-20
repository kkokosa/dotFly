using System;
using Godot;

namespace DotFly.Sample.Godot3D;

/// <summary>
/// Procedural PBR materials (albedo + normal map from a height field, roughness) so the room needs
/// no downloaded assets. Noise comes from Godot's <see cref="FastNoiseLite"/>.
/// </summary>
public static class ProceduralTextures
{
    /// <summary>Oak floor planks with grain, seams and a normal map; world-scale triplanar (planks ≈ 0.8 m wide).</summary>
    public static StandardMaterial3D Wood()
    {
        const int size = 1024;
        var noise = new FastNoiseLite { NoiseType = FastNoiseLite.NoiseTypeEnum.Perlin, Frequency = 0.02f, Seed = 3 };
        var fine = new FastNoiseLite { NoiseType = FastNoiseLite.NoiseTypeEnum.SimplexSmooth, Frequency = 0.15f, Seed = 9 };
        var rng = new Random(1);
        const int planks = 5;
        int plankH = size / planks;
        var tint = new float[planks];
        var shift = new int[planks];
        for (int i = 0; i < planks; i++)
        {
            tint[i] = 0.82f + (float)rng.NextDouble() * 0.36f;
            shift[i] = rng.Next(size);
        }

        var albedo = Image.CreateEmpty(size, size, true, Image.Format.Rgb8);
        var height = Image.CreateEmpty(size, size, false, Image.Format.Rgb8);
        for (int y = 0; y < size; y++)
        {
            int plank = Math.Min(y / plankH, planks - 1);
            int yy = y % plankH;
            bool seam = yy < 3 || yy >= plankH - 2;
            for (int x = 0; x < size; x++)
            {
                int xs = (x + shift[plank]) % size;
                float grain = 0.5f + 0.5f * noise.GetNoise2D(xs * 0.35f, y * 4.5f + plank * 900);     // stretched along the plank
                float rings = 0.5f + 0.5f * MathF.Sin(grain * 14f + fine.GetNoise2D(xs, y) * 2.5f);
                float end = xs % (size / 2) < 3 ? 0.5f : 1f;                                            // plank ends
                float v = (0.5f + 0.22f * grain + 0.12f * rings) * tint[plank] * end;
                if (seam)
                {
                    v *= 0.55f;
                }

                albedo.SetPixel(x, y, new Color(v * 0.95f, v * 0.68f, v * 0.42f));
                float hgt = seam || end < 1f ? 0.2f : 0.55f + 0.25f * rings + 0.15f * grain;
                height.SetPixel(x, y, new Color(hgt, hgt, hgt));
            }
        }

        albedo.GenerateMipmaps();
        return Material(albedo, height, normalStrength: 4f, roughness: 0.45f, uvScale: 0.25f, metallic: 0f, specular: 0.55f);
    }

    /// <summary>Lightly mottled plaster with a fine bump.</summary>
    public static StandardMaterial3D Plaster(Color baseColor)
    {
        const int size = 512;
        var noise = new FastNoiseLite { NoiseType = FastNoiseLite.NoiseTypeEnum.Cellular, Frequency = 0.08f, Seed = 7, FractalOctaves = 3 };
        var albedo = Image.CreateEmpty(size, size, true, Image.Format.Rgb8);
        var height = Image.CreateEmpty(size, size, false, Image.Format.Rgb8);
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                float n = 0.5f + 0.5f * noise.GetNoise2D(x, y);
                float v = 0.94f + 0.06f * n;
                albedo.SetPixel(x, y, new Color(baseColor.R * v, baseColor.G * v, baseColor.B * v));
                height.SetPixel(x, y, new Color(n, n, n));
            }
        }

        albedo.GenerateMipmaps();
        return Material(albedo, height, normalStrength: 1.5f, roughness: 0.92f, uvScale: 0.5f, metallic: 0f, specular: 0.3f);
    }

    /// <summary>Woven upholstery.</summary>
    public static StandardMaterial3D Fabric(Color baseColor)
    {
        const int size = 256;
        var noise = new FastNoiseLite { NoiseType = FastNoiseLite.NoiseTypeEnum.Perlin, Frequency = 0.05f, Seed = 11 };
        var albedo = Image.CreateEmpty(size, size, true, Image.Format.Rgb8);
        var height = Image.CreateEmpty(size, size, false, Image.Format.Rgb8);
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                float weave = ((x / 2 + y / 2) & 1) == 0 ? 1.0f : 0.84f;
                float n = 0.9f + 0.2f * (0.5f + 0.5f * noise.GetNoise2D(x, y));
                float v = weave * n;
                albedo.SetPixel(x, y, new Color(baseColor.R * v, baseColor.G * v, baseColor.B * v));
                height.SetPixel(x, y, new Color(weave, weave, weave));
            }
        }

        albedo.GenerateMipmaps();
        return Material(albedo, height, normalStrength: 1.2f, roughness: 1f, uvScale: 3f, metallic: 0f, specular: 0.15f);
    }

    /// <summary>A striped kilim rug.</summary>
    public static StandardMaterial3D Rug()
    {
        const int size = 512;
        var noise = new FastNoiseLite { NoiseType = FastNoiseLite.NoiseTypeEnum.Perlin, Frequency = 0.1f, Seed = 5 };
        var albedo = Image.CreateEmpty(size, size, true, Image.Format.Rgb8);
        var height = Image.CreateEmpty(size, size, false, Image.Format.Rgb8);
        Color[] stripes = [new(0.5f, 0.16f, 0.14f), new(0.82f, 0.72f, 0.5f), new(0.22f, 0.28f, 0.42f), new(0.82f, 0.72f, 0.5f)];
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                Color c = stripes[(y / 40) % stripes.Length];
                bool border = x < 12 || x >= size - 12 || y < 12 || y >= size - 12;
                bool diamond = ((x / 20 + y / 20) & 1) == 0 && (y / 40) % 2 == 1;
                if (diamond)
                {
                    c = c.Darkened(0.25f);
                }

                float n = 0.88f + 0.24f * (0.5f + 0.5f * noise.GetNoise2D(x, y));
                c = border ? new Color(0.15f, 0.12f, 0.1f) : c * n;
                albedo.SetPixel(x, y, c);
                float w = ((x + y) & 1) == 0 ? 0.6f : 0.4f;
                height.SetPixel(x, y, new Color(w, w, w));
            }
        }

        albedo.GenerateMipmaps();
        return Material(albedo, height, normalStrength: 0.8f, roughness: 1f, uvScale: 1f, metallic: 0f, specular: 0.1f, triplanar: false);
    }

    /// <summary>Wing membrane with veins (alpha = opacity).</summary>
    public static ImageTexture Wing(int size = 128)
    {
        var img = Image.CreateEmpty(size, size, true, Image.Format.Rgba8);
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                float u = x / (float)size, v = y / (float)size;
                float vein = 0;
                foreach (float k in new[] { 0.25f, 0.45f, 0.62f, 0.8f })
                {
                    float d = MathF.Abs(v - (k + 0.12f * MathF.Sin(u * 3f) * (k - 0.5f)));
                    vein = MathF.Max(vein, MathF.Max(0f, 1f - d * 90f));
                }

                foreach (float k in new[] { 0.4f, 0.7f })
                {
                    float d = MathF.Abs(u - k);
                    vein = MathF.Max(vein, MathF.Max(0f, 1f - d * 90f) * (v > 0.25f && v < 0.8f ? 1f : 0f));
                }

                float alpha = 0.22f + 0.65f * vein;
                float tone = 0.9f - 0.5f * vein;
                img.SetPixel(x, y, new Color(tone, tone, tone * 1.05f, alpha));
            }
        }

        img.GenerateMipmaps();
        return ImageTexture.CreateFromImage(img);
    }

    /// <summary>Abdomen stripes (tan with dark bands along the texture's V axis).</summary>
    public static ImageTexture Abdomen(int size = 128)
    {
        var img = Image.CreateEmpty(size, size, true, Image.Format.Rgb8);
        for (int y = 0; y < size; y++)
        {
            float v = y / (float)size;
            float band = MathF.Pow(MathF.Max(0f, MathF.Sin(v * MathF.PI * 5f)), 10f);
            float dark = v > 0.8f ? 0.9f : band * 0.75f;    // the tip of a male abdomen is dark
            var c = new Color(0.78f, 0.55f, 0.28f).Lerp(new Color(0.12f, 0.08f, 0.05f), dark);
            for (int x = 0; x < size; x++)
            {
                img.SetPixel(x, y, c);
            }
        }

        img.GenerateMipmaps();
        return ImageTexture.CreateFromImage(img);
    }

    private static StandardMaterial3D Material(Image albedo, Image height, float normalStrength, float roughness, float uvScale, float metallic, float specular, bool triplanar = true)
    {
        Image normal = (Image)height.Duplicate();
        normal.BumpMapToNormalMap(normalStrength);
        normal.GenerateMipmaps();
        return new StandardMaterial3D
        {
            AlbedoTexture = ImageTexture.CreateFromImage(albedo),
            NormalEnabled = true,
            NormalTexture = ImageTexture.CreateFromImage(normal),
            NormalScale = 1f,
            Roughness = roughness,
            Metallic = metallic,
            MetallicSpecular = specular,
            Uv1Triplanar = triplanar,
            Uv1Scale = new Vector3(uvScale, uvScale, uvScale),
            TextureFilter = BaseMaterial3D.TextureFilterEnum.LinearWithMipmapsAnisotropic,
        };
    }
}
