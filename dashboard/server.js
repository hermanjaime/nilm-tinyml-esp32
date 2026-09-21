const express    = require("express");
const { MongoClient } = require("mongodb");
const cors       = require("cors");
const path       = require("path");

const app  = express();
const PORT = 3000;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

const MONGO_URL = "mongodb://localhost:27017";
const client    = new MongoClient(MONGO_URL);

let collection;

async function start() {
    await client.connect();
    const db   = client.db("nilm_db");
    collection = db.collection("measurements");

    console.log("✅ MongoDB: nilm_db.measurements");

    app.listen(PORT, "0.0.0.0", () => {
        console.log(`✅ API:       http://localhost:${PORT}`);
        console.log(`✅ Dashboard: http://localhost:${PORT}/dashboard`);
    });
}

app.get("/", (req, res) =>
    res.send("NILM API — access /dashboard")
);

app.get("/dashboard", (req, res) =>
    res.sendFile(path.join(__dirname, "public", "dashboard.html"))
);

// POST /measure — accepts three formats:
//   1. Current : { devices[], electrical{} }    ← current NILM_ESP32.ino
//   2. Previous: { devices[], total{} }         ← previous version
//   3. Legacy  : { voltageRms, currentRms, ...} ← original NILM_MongoDB.ino
app.post("/measure", async (req, res) => {
    try {
        const d = req.body;

        // Normalizes the totals block
        // Priority: electrical > total > standalone fields
        const elec = d.electrical || d.total || null;

        const total = elec ? elec : {
            voltageRms      : Number(d.voltageRms)       || 0,
            currentRms      : Number(d.currentRms)       || 0,
            powerW          : Number(d.powerW)            || 0,
            apparentPowerVA : Number(d.apparentPowerVA)   || 0,
            powerFactor     : Number(d.powerFactor)       || 0,
            energyKwh       : Number(d.energyKwh)         || 0,
            costPerKwh      : Number(d.costPerKwh)        || 0,
            totalCost       : Number(d.totalCost)         || 0,
        };

        // Ensures standardized field names in the saved document
        const doc = {
            timestamp  : new Date(),
            clientTime : d.timestamp || null,
            ms         : d.ms        || null,
            source     : d.source    || "esp32",
            devices    : d.devices   || [],

            total      : {
                voltageRms:
                    Number(total.voltageRms) || 0,

                currentRms:
                    Number(total.currentRms) || 0,

                powerW:
                    Number(total.powerW) || 0,

                apparentPowerVA:
                    Number(
                        total.apparentVA ||
                        total.apparentPowerVA
                    ) || 0,

                powerFactor:
                    Number(total.powerFactor) || 0,

                energyKwh:
                    Number(total.energyKwh) || 0,

                costPerKwh:
                    Number(total.costPerKwh) || TARIFA_KWH,

                totalCost:
                    Number(total.totalCost) || 0,
            },
        };

        const result = await collection.insertOne(doc);

        res.status(201).json({
            ok: true,
            insertedId: result.insertedId
        });

    } catch (err) {
        console.error("POST /measure:", err.message);

        res.status(500).json({
            ok: false,
            error: err.message
        });
    }
});

const TARIFA_KWH = 0.95;

app.get("/measurements", async (req, res) => {
    try {
        const limit = Math.min(
            Number(req.query.limit) || 60,
            500
        );

        const data = await collection
            .find({})
            .sort({ timestamp: -1 })
            .limit(limit)
            .toArray();

        res.json(data.reverse());

    } catch (err) {
        res.status(500).json({
            ok: false,
            error: err.message
        });
    }
});

app.get("/latest", async (req, res) => {
    try {
        const doc = await collection
            .find({})
            .sort({ timestamp: -1 })
            .limit(1)
            .toArray();

        res.json(doc[0] || null);

    } catch (err) {
        res.status(500).json({
            ok: false,
            error: err.message
        });
    }
});

start().catch(console.error);