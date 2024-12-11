from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime, timedelta
import json


uri = "mongodb+srv://transyltoonia:meradatabase@transyltoonia.xdrsx1s.mongodb.net/?retryWrites=true&w=majority&appName=transyltoonia"

class DatabaseManager:
    def __init__(self, uri=uri, db_name='FlipkartGrid_DB'):
        """
        Comprehensive Database Management System
        Combines basic and advanced database operations
        
        """
        
        client = MongoClient(uri, server_api=ServerApi('1'))

        try:
            client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            print("An error occurred while connecting to MongoDB: ", e)
        try:
            # Database Connection
            self.client = MongoClient(uri)
            self.db = self.client[db_name]
            # Collections
            self.brand_collection = self.db['brand_recognition']
            self.ocr_collection = self.db['ocr']
            self.freshness_collection = self.db['freshness']
            
            # Create performance indexes
            self._create_indexes()
        except Exception as e:
            print(f"Database connection error: {e}")
            raise

    def _create_indexes(self):
        """
        Create performance and unique indexes
        """
        self.brand_collection.create_index([("brand", 1)], unique=True)
        self.ocr_collection.create_index([("brand", 1), ("expiry_date", 1)])
        self.freshness_collection.create_index([("produce", 1)])

    # Original Basic Methods
    from datetime import datetime

    def add_brand_record(self, brand, count):
        """
        Add or update a brand record with validation.
        If the brand exists, increment its count. Otherwise, add a new record.
        """
        try:
            # Input validation
            if not brand or not isinstance(brand, str):
                return "Invalid brand name"
            if not isinstance(count, (int, float)) or count < 0:
                return "Invalid count value"

            # Prepare the brand name
            brand = brand.strip().title()

            # Check if the brand already exists
            existing_record = self.brand_collection.find_one({"brand": brand})

            if existing_record:
                # Increment the count if the brand exists
                new_count = existing_record["count"] + count
                self.brand_collection.update_one(
                    {"brand": brand},
                    {"$set": {"count": new_count, "last_updated": datetime.now()}}
                )
                return f"Updated brand record: Brand = {brand}, New Count = {new_count}"
            else:
                # Add a new record if the brand does not exist
                record = {
                    "S.No": self.brand_collection.count_documents({}) + 1,
                    "timestamp": datetime.now(),
                    "brand": brand,
                    "count": count,
                    "last_updated": datetime.now()
                }
                self.brand_collection.insert_one(record)
                return f"Added brand record: Brand = {brand}, Count = {count}"
        except Exception as e:
            return f"Error adding brand record: {e}"


    def get_brand_records(self, filter_criteria=None, sort_by='timestamp', ascending=False):
        """
        Retrieve brand records with flexible filtering and sorting
        """
        try:
            filter_criteria = filter_criteria or {}
            sort_direction = 1 if ascending else -1
            
            records = self.brand_collection.find(filter_criteria).sort(sort_by, sort_direction)
            
            return [
                {
                    "S.No": record['S.No'],
                    "Brand": record['brand'],
                    "Count": record['count'],
                    "Timestamp": record['timestamp']
                } for record in records
            ]
        except Exception as e:
            return f"Error retrieving brand records: {e}"

    def add_ocr_record(self, brand, expiry_date, manufacture_date, mrp):
        """
        Add OCR record with comprehensive validation
        """
        try:
            # Date validation helper
            def validate_date(date_str):
                try:
                    return datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    return None

            exp_date = validate_date(expiry_date)
            man_date = validate_date(manufacture_date)

            if not exp_date or not man_date:
                return "Invalid date format. Use YYYY-MM-DD"

            # Check expiry date is after manufacture date
            if exp_date <= man_date:
                return "Expiry date must be after manufacture date"

            # Validate MRP
            try:
                mrp = float(mrp)
                if mrp <= 0:
                    return "MRP must be a positive number"
            except ValueError:
                return "Invalid MRP value"

            record = {
                "S.No": self.ocr_collection.count_documents({}) + 1,
                "timestamp": datetime.now(),
                "brand": brand.strip().title(),
                "expiry_date": exp_date,
                "manufacture_date": man_date,
                "mrp": mrp,
                "days_to_expiry": (exp_date - datetime.now().date()).days
            }
            self.ocr_collection.insert_one(record)
            return f"Added OCR record: Brand = {brand}, Expiry Date = {expiry_date}"
        except Exception as e:
            return f"Error adding OCR record: {e}"
        
        
    def get_ocr_records(self, filter_criteria=None, sort_by='timestamp', ascending=False):
        """
        Retrieve OCR records with flexible filtering and sorting
        """
        try:
            filter_criteria = filter_criteria or {}
            sort_direction = 1 if ascending else -1
            
            records = self.ocr_collection.find(filter_criteria).sort(sort_by, sort_direction)
            
            return [
                {
                    "S.No": record['S.No'],
                    "Brand": record['brand'],
                    "Expiry Date": record['expiry_date'],
                    "Manufacture Date": record['manufacture_date'],
                    "MRP": record['mrp']
                } for record in records
            ]
        except Exception as e:
            return f"Error retrieving OCR records: {e}"

    
    def add_freshness_record(self, produce, shelf_life, characteristics, eatable):
        """
        Add freshness record with enhanced validation
        """
        try:
            # Input validations
            if not produce or not isinstance(produce, str):
                return "Invalid produce name"
            
            # Convert to boolean or validate eatable status
            if isinstance(eatable, str):
                eatable = eatable.lower() in ['true', 'yes', '1']
            
            record = {
                "S.No": self.freshness_collection.count_documents({}) + 1,
                "timestamp": datetime.now(),
                "produce": produce.strip().title(),
                "Shelf-Life": shelf_life,
                "Characteristics": characteristics,
                "Eatable": bool(eatable)
            }
            self.freshness_collection.insert_one(record)
            return f"Added freshness record: Produce = {produce}, Shelf-Life = {shelf_life}"
        except Exception as e:
            return f"Error adding freshness record: {e}"

    # Advanced Analysis Methods
    def analyze_brand_trends(self, time_period=30):
        """
        Analyze brand trends over a specified time period
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=time_period)
            
            pipeline = [
                {"$match": {"timestamp": {"$gte": cutoff_date}}},
                {"$group": {
                    "_id": "$brand",
                    "total_count": {"$sum": "$count"},
                    "avg_count": {"$avg": "$count"},
                    "first_seen": {"$min": "$timestamp"},
                    "last_seen": {"$max": "$timestamp"}
                }},
                {"$sort": {"total_count": -1}}
            ]
            
            trends = list(self.brand_collection.aggregate(pipeline))
            
            # Enrich the results
            for trend in trends:
                trend['brand'] = trend.pop('_id')
                trend['first_seen'] = trend['first_seen'].strftime('%Y-%m-%d %H:%M:%S')
                trend['last_seen'] = trend['last_seen'].strftime('%Y-%m-%d %H:%M:%S')
                trend['total_count'] = round(trend['total_count'], 2)
                trend['avg_count'] = round(trend['avg_count'], 2)
            
            return trends
        except Exception as e:
            return f"Error analyzing brand trends: {e}"

    # Additional methods from previous implementations
    def get_all_brand_records(self):
        """
        Retrieve all brand records
        """
        try:
            records = self.brand_collection.find()
            result = [
                f"S.No: {record['S.No']}, Brand: {record['brand']}, Count: {record['count']}"
                for record in records
            ]
            return "\n".join(result) if result else "No brand records found."
        except Exception as e:
            return f"An error occurred while fetching brand records: {e}"

    def get_all_ocr_records(self):
        """
        Retrieve all OCR records
        """
        try:
            records = self.ocr_collection.find()
            result = [
                f"S.No: {record['S.No']}, Brand: {record['brand']}, Expiry Date: {record['expiry_date']}, "
                f"Manufacture Date: {record['manufacture_date']}, MRP: {record['mrp']}"
                for record in records
            ]
            return "\n".join(result) if result else "No OCR records found."
        except Exception as e:
            return f"An error occurred while fetching OCR records: {e}"

    def get_all_freshness_records(self):
        """
        Retrieve all freshness records
        """
        try:
            records = self.freshness_collection.find()
            result = [
                f"S.No: {record['S.No']}, Produce: {record['produce']}, Shelf-Life: {record['Shelf-Life']}, "
                f"Characteristics: {record['Characteristics']}, Eatable: {record['Eatable']}"
                for record in records
            ]
            return "\n".join(result) if result else "No freshness records found."
        except Exception as e:
            return f"An error occurred while fetching freshness records: {e}"
       
    def get_freshness_records(self, filter_criteria=None, sort_by='timestamp', ascending=False):
        """
        Retrieve freshness records with flexible filtering and sorting
        """
        try:
            filter_criteria = filter_criteria or {}
            sort_direction = 1 if ascending else -1
            
            records = self.freshness_collection.find(filter_criteria).sort(sort_by, sort_direction)
            
            return [
                {
                    "S.No": record['S.No'],
                    "Produce": record['produce'],
                    "Shelf-Life": record['Shelf-Life'],
                    "Characteristics": record['Characteristics'],
                    "Eatable": record['Eatable']
                } for record in records
            ]
        except Exception as e:
            return f"Error retrieving freshness records: {e}"
        
    

    # Advanced Search and Filtering
    def advanced_search(self, collection_name, search_criteria=None, projection=None):
        """
        Advanced search method with flexible filtering
        """
        try:
            collection_map = {
                'brand': self.brand_collection,
                'ocr': self.ocr_collection,
                'freshness': self.freshness_collection
            }
            
            if collection_name not in collection_map:
                return "Invalid collection name"
            
            search_criteria = search_criteria or {}
            projection = projection or {}
            
            results = list(collection_map[collection_name].find(search_criteria, projection))
            
            # Convert ObjectId to string for JSON serialization
            for result in results:
                if '_id' in result:
                    result['_id'] = str(result['_id'])
            
            return results
        except Exception as e:
            return f"Error performing advanced search: {e}"

    # More Advanced Methods (Export, Statistical Analysis, etc.)
    def export_collection_to_json(self, collection_name, filename=None):
        """
        Export a collection to a JSON file
        """
        try:
            collection_map = {
                'brand': self.brand_collection,
                'ocr': self.ocr_collection,
                'freshness': self.freshness_collection
            }
            
            if collection_name not in collection_map:
                return "Invalid collection name"
            
            documents = list(collection_map[collection_name].find())
            
            for doc in documents:
                doc['_id'] = str(doc['_id'])
            
            if not filename:
                filename = f"{collection_name}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w') as f:
                json.dump(documents, f, indent=2, default=str)
            
            return f"Successfully exported {len(documents)} documents to {filename}"
        except Exception as e:
            return f"Error exporting collection: {e}"

    def close_connection(self):
        """
        Properly close the MongoDB connection
        """
        try:
            self.client.close()
            print("MongoDB connection closed successfully")
        except Exception as e:
            print(f"Error closing connection: {e}")

